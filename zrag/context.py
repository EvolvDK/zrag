"""
Context management for zrag.
"""

import json
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ContextNode:
    path: str
    description: str
    children: Dict[str, 'ContextNode'] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())


class ContextManager:
    """Manages hierarchical context trees."""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.context_file = data_dir / "contexts.json"
        self._contexts: Dict[str, ContextNode] = {}
        self._load_contexts()
    
    def _load_contexts(self):
        if self.context_file.exists():
            try:
                with open(self.context_file, 'r') as f:
                    data = json.load(f)
                    for path, context_data in data.items():
                        self._contexts[path] = ContextNode(
                            path=context_data['path'],
                            description=context_data['description'],
                            children={},
                            created_at=context_data.get('created_at', 0),
                            updated_at=context_data.get('updated_at', 0),
                        )
            except Exception as e:
                print(f"⚠ Failed to load contexts: {e}")
    
    def _save_contexts(self):
        try:
            self.context_file.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for path, node in self._contexts.items():
                data[path] = {
                    'path': node.path,
                    'description': node.description,
                    'created_at': node.created_at,
                    'updated_at': node.updated_at,
                }
            with open(self.context_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠ Failed to save contexts: {e}")
    
    def add_context(self, path: str, description: str) -> ContextNode:
        if path in self._contexts:
            self._contexts[path].description = description
            self._contexts[path].updated_at = datetime.now().timestamp()
        else:
            self._contexts[path] = ContextNode(path=path, description=description)
        self._save_contexts()
        return self._contexts[path]
    
    def remove_context(self, path: str) -> int:
        """Remove contexts matching a path or glob pattern. Returns the number of removed contexts."""
        removed_count = 0
        
        if "*" in path or "?" in path:
            keys_to_remove =[k for k in self._contexts.keys() if fnmatch.fnmatch(k, path)]
            for k in keys_to_remove:
                del self._contexts[k]
                removed_count += 1
        else:
            if path in self._contexts:
                del self._contexts[path]
                removed_count += 1
                
        if removed_count > 0:
            self._save_contexts()
            
        return removed_count
    
    def get_context(self, path: str) -> Optional[ContextNode]:
        return self._contexts.get(path)
    
    def list_contexts(self, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        contexts =[]
        for path, node in self._contexts.items():
            if collection_name and not path.startswith(collection_name):
                continue
            contexts.append({
                'path': path,
                'description': node.description,
                'created_at': node.created_at,
                'updated_at': node.updated_at,
            })
        return contexts
    
    def resolve_context(self, source_path: str) -> List[str]:
        contexts =[]
        if source_path in self._contexts:
            contexts.append(self._contexts[source_path].description)
        parts = source_path.split('/')
        for i in range(len(parts) - 1, 0, -1):
            parent_path = '/'.join(parts[:i])
            if parent_path in self._contexts:
                contexts.append(self._contexts[parent_path].description)
        if 'global' in self._contexts:
            contexts.append(self._contexts['global'].description)
        return contexts

    def check_missing_context(self, source_paths: List[str]) -> List[str]:
        """Check which source paths lack context definitions.

        Args:
            source_paths: List of source file paths to check

        Returns:
            List of paths that don't have any context (direct or parent)
        """
        missing = []
        for source_path in source_paths:
            # Check if path has direct context
            if source_path in self._contexts:
                continue

            # Check if any parent path has context
            parts = source_path.split('/')
            has_parent_context = False
            for i in range(len(parts) - 1, 0, -1):
                parent_path = '/'.join(parts[:i])
                if parent_path in self._contexts:
                    has_parent_context = True
                    break

            # Check if global context exists
            has_global_context = 'global' in self._contexts

            # Path is missing if it has no direct, parent, or global context
            if not has_parent_context and not has_global_context:
                missing.append(source_path)

        return missing

    def clear_contexts(self):
        """Clear all contexts (useful for test cleanup)."""
        self._contexts.clear()
        self._save_contexts()
