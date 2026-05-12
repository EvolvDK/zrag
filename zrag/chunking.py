"""
Chunking strategies for zrag.
"""

import tiktoken
import re
from pathlib import Path
from typing import List, Tuple, Optional

from tree_sitter import Language, Parser
import tree_sitter_python as ts_python
import tree_sitter_javascript as ts_javascript
import tree_sitter_rust as ts_rust
import tree_sitter_go as ts_go
import tree_sitter_typescript as ts_typescript
import tree_sitter_cpp as ts_cpp

_TS_LANGUAGES = None
_ENC = None

def get_ts_language(lang: str) -> Optional[Language]:
    global _TS_LANGUAGES
    if _TS_LANGUAGES is None:
        try:
            _TS_LANGUAGES = {
                'python': Language(ts_python.language()),
                'javascript': Language(ts_javascript.language()),
                'rust': Language(ts_rust.language()),
                'go': Language(ts_go.language()),
                'typescript': Language(ts_typescript.language_typescript()),
                'cpp': Language(ts_cpp.language()),
            }
        except Exception as e:
            print(f"⚠ Tree-sitter initialization failed: {e}")
            _TS_LANGUAGES = {}
    return _TS_LANGUAGES.get(lang)

def get_encoding():
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC

def chunk_text_by_tokens(text: str, max_tokens: int, overlap: float = 0.15) -> List[str]:
    if not text.strip():
        return []
    enc = get_encoding()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text.strip()]

    overlap_tokens = int(max_tokens * overlap)
    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        if end_idx == len(tokens):
            break
        start_idx = end_idx - overlap_tokens
        if start_idx >= end_idx:
            start_idx = end_idx
    return [c for c in chunks if c.strip()]


def chunk_text_simple(text: str, max_tokens: int = 900, overlap: float = 0.15) -> List[str]:
    return chunk_text_by_tokens(text, max_tokens, overlap)


def chunk_code_ast(filepath: Path, lang: str, max_tokens: int) -> List[Tuple[str, str]]:
    ts_lang = get_ts_language(lang)
    with open(filepath, 'r', encoding='utf-8') as f:
        source_code = f.read()

    if not ts_lang:
        return[(source_code, 'full_file')]

    parser = Parser(ts_lang)
    tree = parser.parse(bytes(source_code, "utf8"))
    chunks =[]
    enc = get_encoding()

    def traverse(node):
        node_text = source_code[node.start_byte:node.end_byte].strip()
        significant_nodes = {
            'class_definition', 'function_definition', 'decorated_definition',
            'class_declaration', 'function_declaration', 'method_definition',
            'arrow_function', 'interface_declaration', 'type_alias_declaration',
            'method_declaration', 'type_declaration',
            'function_item', 'struct_item', 'enum_item', 'trait_item', 'impl_item', 'mod_item',
            'class_specifier', 'struct_specifier', 'enum_specifier',
            'namespace_definition', 'template_declaration', 'function_declarator', 'constructor_definition',
            'module', 'import_statement', 'export_statement', 'declaration'
        }

        if node.type in significant_nodes:
            if len(enc.encode(node_text)) > max_tokens * 1.5:
                for child in node.children:
                    traverse(child)
            else:
                chunks.append((node_text, node.type))
                return
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    if chunks:
        return chunks
    
    fallback_chunks = chunk_text_by_tokens(source_code, max_tokens, overlap=0.15)
    return[(chunk, 'full_file_chunked') for chunk in fallback_chunks]


def chunk_text_semantic(text: str, max_tokens: int, overlap: float) -> List[str]:
    enc = get_encoding()
    window_tokens = max(1, int(max_tokens * overlap))
    
    code_blocks =[(m.start(), m.end()) for m in re.finditer(r'```.*?```', text, flags=re.DOTALL)]
    def is_protected(idx):
        return any(start < idx < end for start, end in code_blocks)

    bp_regex = re.compile(
        r'^(?P<h1>#\s+)|^(?P<h2>##\s+)|^(?P<h3>###\s+)|^(?P<h4>####\s+)|'
        r'^(?P<h5>#####\s+)|^(?P<h6>######\s+)|^(?P<code>```)|^(?P<hr>(?:---|[*]{3,})\s*)|'
        r'(?P<para>\n\n+)|^(?P<list>(?:[-*]|\d+\.)\s+)|(?P<line>\n)',
        flags=re.MULTILINE
    )
    
    scores = {
        'h1': 100, 'h2': 90, 'h3': 80, 'h4': 70, 'h5': 60, 'h6': 50,
        'code': 80, 'hr': 60, 'para': 20, 'list': 5, 'line': 1
    }
    
    breakpoints =[]
    for match in bp_regex.finditer(text):
        bp_type = match.lastgroup
        idx = match.start()
        if not is_protected(idx):
            breakpoints.append((idx, scores[bp_type]))

    breakpoints.append((len(text), 1000))
    
    chunks =[]
    current_start = 0
    start_token_idx = 0
    tokens = enc.encode(text)
    
    while current_start < len(text):
        target_token_idx = start_token_idx + max_tokens
        
        if target_token_idx >= len(tokens):
            chunks.append(text[current_start:].strip())
            break
            
        prefix_bytes = enc.decode_bytes(tokens[:target_token_idx])
        target_char_idx = len(prefix_bytes.decode('utf-8', errors='ignore'))
        
        window_start_token_idx = max(start_token_idx, target_token_idx - window_tokens)
        window_bytes = enc.decode_bytes(tokens[:window_start_token_idx])
        window_start_char_idx = len(window_bytes.decode('utf-8', errors='ignore'))
        
        valid_bps =[bp for bp in breakpoints if window_start_char_idx <= bp[0] <= target_char_idx and bp[0] > current_start]
        
        best_bp = None
        best_score = -1
        
        if valid_bps:
            for idx, base_score in valid_bps:
                dist_tokens = len(enc.encode(text[idx:target_char_idx]))
                dist_tokens = min(dist_tokens, window_tokens)
                penalty = (dist_tokens / window_tokens) ** 2 * 0.7
                final_score = base_score * (1 - penalty)
                
                if final_score > best_score:
                    best_score = final_score
                    best_bp = idx
        
        if best_bp is not None:
            added_text = text[current_start:best_bp]
            current_start = best_bp
        else:
            added_text = text[current_start:target_char_idx]
            current_start = target_char_idx

        if added_text.strip():
            chunks.append(added_text.strip())
            
        start_token_idx += len(enc.encode(added_text))
            
    return[c for c in chunks if c.strip()]


def get_file_type(filepath: Path) -> str:
    ext = filepath.suffix.lower()
    type_map = {
        '.md': 'markdown', '.pdf': 'pdf', '.py': 'code_python',
        '.js': 'code_javascript', '.rs': 'code_rust', '.go': 'code_go',
        '.ts': 'code_typescript', '.tsx': 'code_typescript', '.cpp': 'code_cpp',
        '.hpp': 'code_cpp', '.cc': 'code_cpp', '.cxx': 'code_cpp',
        '.h': 'code_cpp', '.png': 'image', '.jpg': 'image', '.jpeg': 'image',
        '.docx': 'docx',
    }
    return type_map.get(ext, 'unknown')
