"""
Shared fixtures for zrag test suite.

All tests use isolated tmp dirs — no ~/.zrag pollution.
Engine fixtures use local dense embeddings (no GPU required for smoke tests).
"""

import os
import shutil
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Generator

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PYTHON_SAMPLE = textwrap.dedent("""\
    class AuthManager:
        \"\"\"Handles user authentication.\"\"\"

        def __init__(self, secret: str):
            self.secret = secret
            self._sessions: dict = {}

        def login(self, user_id: str, password: str) -> str:
            \"\"\"Validate credentials and return session token.\"\"\"
            if not self._validate(user_id, password):
                raise ValueError("Invalid credentials")
            token = self._generate_token(user_id)
            self._sessions[token] = user_id
            return token

        def logout(self, token: str) -> None:
            \"\"\"Invalidate session token.\"\"\"
            self._sessions.pop(token, None)

        def _validate(self, user_id: str, password: str) -> bool:
            return bool(user_id) and bool(password)

        def _generate_token(self, user_id: str) -> str:
            import hashlib, time
            return hashlib.sha256(f"{user_id}{time.time()}{self.secret}".encode()).hexdigest()


    def hash_password(password: str, salt: str = "") -> str:
        \"\"\"Hash a password with optional salt.\"\"\"
        import hashlib
        return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()


    def verify_token(token: str, sessions: dict) -> bool:
        \"\"\"Check if a token is valid.\"\"\"
        return token in sessions
""")

MARKDOWN_SAMPLE = textwrap.dedent("""\
    # Authentication Guide

    This guide explains how authentication works in our system.

    ## Overview

    Authentication uses JWT tokens with a 24-hour expiry window.
    Tokens are stored server-side and validated on each request.

    ## Login Flow

    1. User submits credentials via POST /auth/login
    2. Server validates username and password hash
    3. On success, a signed JWT is returned
    4. Client stores JWT in memory (never in localStorage)

    ## Token Validation

    Every protected endpoint calls `verify_token(token)` before processing.

    ```python
    def verify_token(token: str) -> bool:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload.get("sub") is not None
    ```

    ## Logout

    Send DELETE /auth/logout with Authorization header.
    Server immediately invalidates the token.

    ---

    ## Security Considerations

    ### Password Hashing

    Passwords are hashed with bcrypt (cost factor 12).
    Never store plaintext passwords. Always use `hash_password()`.

    ### Rate Limiting

    Login attempts are rate-limited to 5 per minute per IP.
    After 10 failures, the account is temporarily locked.

    ### Session Expiry

    Sessions expire after 24 hours of inactivity.
    Refresh tokens extend the window by another 24 hours.
""")

GO_SAMPLE = textwrap.dedent("""\
    package auth

    import (
        "crypto/sha256"
        "fmt"
        "sync"
        "time"
    )

    // SessionStore manages active sessions.
    type SessionStore struct {
        mu       sync.RWMutex
        sessions map[string]Session
    }

    // Session holds user session data.
    type Session struct {
        UserID    string
        ExpiresAt time.Time
    }

    // NewSessionStore creates a new store.
    func NewSessionStore() *SessionStore {
        return &SessionStore{sessions: make(map[string]Session)}
    }

    // Create adds a new session and returns the token.
    func (s *SessionStore) Create(userID string, ttl time.Duration) string {
        token := generateToken(userID)
        s.mu.Lock()
        s.sessions[token] = Session{UserID: userID, ExpiresAt: time.Now().Add(ttl)}
        s.mu.Unlock()
        return token
    }

    // Validate checks if a token is valid and not expired.
    func (s *SessionStore) Validate(token string) (string, bool) {
        s.mu.RLock()
        defer s.mu.RUnlock()
        sess, ok := s.sessions[token]
        if !ok || time.Now().After(sess.ExpiresAt) {
            return "", false
        }
        return sess.UserID, true
    }

    func generateToken(userID string) string {
        h := sha256.Sum256([]byte(fmt.Sprintf("%s%d", userID, time.Now().UnixNano())))
        return fmt.Sprintf("%x", h)
    }
""")

RUST_SAMPLE = textwrap.dedent("""\
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};

    #[derive(Debug, Clone)]
    pub struct Session {
        pub user_id: String,
        pub expires_at: u64,
    }

    pub struct SessionStore {
        sessions: Arc<RwLock<HashMap<String, Session>>>,
    }

    impl SessionStore {
        pub fn new() -> Self {
            SessionStore {
                sessions: Arc::new(RwLock::new(HashMap::new())),
            }
        }

        pub fn create(&self, user_id: String, ttl_secs: u64) -> String {
            let token = format!("{}-{}", user_id, ttl_secs);
            let session = Session { user_id, expires_at: ttl_secs };
            self.sessions.write().unwrap().insert(token.clone(), session);
            token
        }

        pub fn validate(&self, token: &str) -> Option<String> {
            let sessions = self.sessions.read().unwrap();
            sessions.get(token).map(|s| s.user_id.clone())
        }

        pub fn remove(&self, token: &str) -> bool {
            self.sessions.write().unwrap().remove(token).is_some()
        }
    }

    pub fn hash_password(password: &str) -> String {
        format!("hashed_{}", password)
    }
""")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tmp_base() -> Generator[Path, None, None]:
    """Single temp base dir reused across session."""
    d = Path(tempfile.mkdtemp(prefix="zrag_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Per-test isolated tmp dir (pytest built-in tmp_path)."""
    return tmp_path


@pytest.fixture(scope="session")
def zrag_config(tmp_base: Path):
    """Isolated Config pointing at tmp dirs. Session-scoped for efficiency."""
    from zrag.config import Config
    cfg = Config(
        data_dir=tmp_base / "data",
        collections_dir=tmp_base / "collections",
        dense_embedding_type="local",
        top_k=5,
        reranker_type="none",  # skip heavy reranker in unit tests
    )
    cfg.ensure_directories()
    return cfg


@pytest.fixture(scope="session")
def engine(zrag_config):
    """ZragEngine with isolated config. Session-scoped to reuse across tests."""
    from zrag.core import ZragEngine
    eng = ZragEngine(zrag_config)
    yield eng
    # Cleanup: flush all collections (zvec doesn't have close() method)
    for name in list(eng._collections.keys()):
        try:
            eng._collections[name].flush()
        except Exception:
            pass
    eng._collections.clear()


@pytest.fixture
def collection(engine):
    """Engine + pre-created 'test' collection. Function-scoped for test isolation."""
    # Remove collection if it already exists from a previous failed test
    try:
        engine.remove_collection("test", force=True)
    except Exception:
        pass

    engine.create_collection("test")
    yield engine
    # Cleanup: remove collection after test
    try:
        engine.remove_collection("test", force=True)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def clear_contexts_between_tests(engine):
    """Clear contexts before each test to ensure test isolation."""
    engine.context_manager.clear_contexts()
    yield


@pytest.fixture
def populated_collection(collection, tmp_path: Path):
    """Collection pre-loaded with Python + Markdown samples."""
    py_file = tmp_path / "auth.py"
    py_file.write_text(PYTHON_SAMPLE)
    md_file = tmp_path / "guide.md"
    md_file.write_text(MARKDOWN_SAMPLE)

    collection.ingest_file("test", py_file)
    collection.ingest_file("test", md_file)
    yield collection
    # Cleanup handled by parent collection fixture


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    """Fake source repo with mixed files for update/sync tests."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text(PYTHON_SAMPLE)
    (repo / "guide.md").write_text(MARKDOWN_SAMPLE)
    (repo / "session.go").write_text(GO_SAMPLE)
    (repo / "store.rs").write_text(RUST_SAMPLE)
    (repo / "readme.txt").write_text("Simple readme file for testing purposes.\n")
    return repo


@pytest.fixture(scope="session")
def sdk(engine):
    """ZragSDK wrapping the test engine. Session-scoped for efficiency."""
    from zrag.sdk import ZragSDK
    return ZragSDK(engine)


# ---------------------------------------------------------------------------
# Sample text helpers (reusable in tests)
# ---------------------------------------------------------------------------

def make_python_file(path: Path, content: str = PYTHON_SAMPLE) -> Path:
    p = path / "sample.py"
    p.write_text(content)
    return p


def make_markdown_file(path: Path, content: str = MARKDOWN_SAMPLE) -> Path:
    p = path / "sample.md"
    p.write_text(content)
    return p
