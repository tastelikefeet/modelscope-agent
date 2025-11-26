"""Multi-language Import Parser

This module provides utilities to parse import/include statements from various
programming languages and extract detailed information about dependencies.

Supported Languages:
- JavaScript/TypeScript (.js, .ts, .jsx, .tsx, .mjs, .cjs)
- Python (.py)
- C/C++ (.c, .cpp, .cc, .cxx, .h, .hpp)
- Rust (.rs)
- Java/Kotlin (.java, .kt, .kts)
- Go (.go)

Main Functions:

1. parse_imports_detailed(file_path, code_content) -> List[ImportInfo]
   Returns detailed information including:
   - source_file: Resolved file path
   - imported_items: List of imported symbols/classes/functions
   - import_type: 'named', 'default', 'namespace', or 'side-effect'
   - alias: Import alias if any
   - is_type_only: Whether it's a type-only import (TypeScript)
   - raw_statement: Original import statement

2. parse_imports(file_path, code_content) -> List[str]
   Backward compatible function that returns only file paths

Example Usage:

    from utils import parse_imports_detailed
    
    code = '''
    import { User, UserRole } from '../models/User';
    import * as utils from './utils';
    import type { Config } from './config';
    '''
    
    imports = parse_imports_detailed('src/index.ts', code)
    
    for imp in imports:
        print(f"File: {imp.source_file}")
        print(f"Items: {imp.imported_items}")
        print(f"Type: {imp.import_type}")
        if imp.is_type_only:
            print("  (Type-only import)")

Key Features:
- Handles multi-line import statements
- Extracts specific imported items (classes, functions, variables)
- Resolves relative file paths
- Distinguishes between different import types
- Filters out external packages (npm, pip, etc.)
"""

import os
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field


@dataclass
class ImportInfo:
    """Detailed information about an import statement"""
    # Source file path (resolved path)
    source_file: str
    # Original import statement
    raw_statement: str
    # What's being imported (e.g., ['User', 'UserRole'] or ['*'] or ['default'])
    imported_items: List[str] = field(default_factory=list)
    # Import type: 'named', 'default', 'namespace', 'side-effect'
    import_type: str = 'named'
    # Alias if any (e.g., 'import * as utils' -> 'utils')
    alias: Optional[str] = None
    # Whether this is a type-only import (TypeScript)
    is_type_only: bool = False

    def __repr__(self):
        items_str = ', '.join(self.imported_items) if self.imported_items else 'all'
        alias_str = f' as {self.alias}' if self.alias else ''
        return f"ImportInfo(file='{self.source_file}', items=[{items_str}]{alias_str})"


stop_words = [
    "\nclass ",
    "\ndef ",
    "\nfunc ",
    "\nfunction ",
    "\npub fn ",
    "\nfn ",
    "\nstruct ",
    "\nenum ",
    "\nexport ",
    "\ninterface ",
    "\ntrait ",
    "\nimpl ",
    "\nmodule ",
    "\ntype ",
    "\npublic class ",
    "\nprivate class ",
    "\nprotected class ",
    "\npublic interface ",
    "\npublic enum ",
    "\npublic struct ",
    "\nabstract class ",
    "\nconst ",
    "\nlet ",
    "\nvar ",
    "\nasync def ",
    "\n@",
]


def parse_imports_detailed(current_file: str, code_content: str) -> List[ImportInfo]:
    """Parse imports and return detailed information about what's imported from each file"""
    imports = []
    current_dir = os.path.dirname(current_file) if current_file else '.'

    # Detect file extension
    file_ext = os.path.splitext(current_file)[1].lstrip('.').lower() if current_file else ''

    # Import patterns for different languages with detailed extraction
    # Pattern structure: (regex_pattern, file_extensions, resolver_function, regex_flags)
    patterns = [
        # Python: from ... import ...
        (r'^\s*from\s+([\w.]+)\s+import\s+([^\n]+)', ['py'], _extract_python_import, re.MULTILINE),
        # Python: import ...
        (r'^\s*import\s+([\w.,\s]+)', ['py'], _extract_python_import_simple, re.MULTILINE),

        # JavaScript/TypeScript: import ... from '...'
        # Match: import { A, B } from 'path' or import A from 'path' or import * as A from 'path'
        (r"import\s+(type\s+)?(?:(\{[^}]*\}|\*\s+as\s+\w+|\w+)\s*,?\s*)*from\s+['\"]([^'\"]+)['\"]",
         ['js', 'ts', 'jsx', 'tsx', 'mjs', 'cjs'], _extract_js_import, re.MULTILINE | re.DOTALL),
        # Match: import 'path' (side-effect)
        (r"^\s*import\s+['\"]([^'\"]+)['\"]", ['js', 'ts', 'jsx', 'tsx', 'mjs', 'cjs'],
         _extract_js_side_effect, re.MULTILINE),
        # Match: export ... from 'path'
        (r"export\s+(type\s+)?(?:(\{[^}]*\}|\*(?:\s+as\s+\w+)?)\s+)?from\s+['\"]([^'\"]+)['\"]",
         ['js', 'ts', 'jsx', 'tsx', 'mjs', 'cjs'], _extract_js_export, re.MULTILINE | re.DOTALL),

        # C/C++: #include
        (r'^\s*#include\s+"([^"]+)"', ['c', 'cpp', 'cc', 'cxx', 'h', 'hpp'],
         _extract_c_include, re.MULTILINE),

        # Rust: use
        (r'^\s*use\s+(?:crate::)?([\w:]+)(?:::\{([^}]+)\})?', ['rs'], _extract_rust_use, re.MULTILINE),

        # Java/Kotlin: import
        (r'^\s*import\s+(static\s+)?([\w.]+(?:\.\*)?)', ['java', 'kt', 'kts'],
         _extract_java_import, re.MULTILINE),

        # Go: import
        (r'^\s*import\s+(?:(\w+)\s+)?"([^"]+)"', ['go'], _extract_go_import, re.MULTILINE),
    ]

    # Process each pattern on the entire content
    for pattern, extensions, extractor, flags in patterns:
        # Skip if file extension doesn't match
        if file_ext and file_ext not in extensions:
            continue

        # Find all matches in the entire content
        for match in re.finditer(pattern, code_content, flags):
            import_info = extractor(match, current_dir, current_file, code_content)
            if import_info:
                if isinstance(import_info, list):
                    imports.extend(import_info)
                else:
                    imports.append(import_info)

    return imports


def parse_imports(current_file: str, code_content: str) -> List[ImportInfo]:
    """Parse imports and return list of file paths (for backward compatibility)"""
    return parse_imports_detailed(current_file, code_content)


# ============================================================================
# Detailed Import Extractors (return ImportInfo objects)
# ============================================================================

def _extract_js_import(match, current_dir, current_file, code_content):
    """Extract JavaScript/TypeScript import details"""
    full_match = match.group(0)
    is_type_only = match.group(1) is not None  # 'type' keyword
    import_path = match.group(3)
    
    # Skip external packages
    if not import_path.startswith('.') and not import_path.startswith('/'):
        return None
    
    # Resolve file path
    source_file = _resolve_js_path(import_path, current_dir)
    
    # Extract imported items
    imported_items = []
    import_type = 'named'
    alias = None
    
    # Parse the import clause (everything before 'from')
    import_clause = full_match.split('from')[0].replace('import', '').strip()
    if is_type_only:
        import_clause = import_clause.replace('type', '').strip()
    
    if import_clause.startswith('{') and import_clause.endswith('}'):
        # Named imports: import { A, B as C } from '...'
        import_type = 'named'
        items_str = import_clause[1:-1]  # Remove { }
        for item in items_str.split(','):
            item = item.strip()
            if ' as ' in item:
                original, alias_name = item.split(' as ')
                imported_items.append(original.strip())
            elif item:
                imported_items.append(item)
    elif import_clause.startswith('*'):
        # Namespace import: import * as name from '...'
        import_type = 'namespace'
        imported_items = ['*']
        if ' as ' in import_clause:
            alias = import_clause.split(' as ')[1].strip()
    elif import_clause:
        # Default import: import Name from '...'
        import_type = 'default'
        imported_items = [import_clause.split(',')[0].strip()]
    
    return ImportInfo(
        source_file=source_file,
        raw_statement=full_match,
        imported_items=imported_items,
        import_type=import_type,
        alias=alias,
        is_type_only=is_type_only
    )


def _extract_js_side_effect(match, current_dir, current_file, code_content):
    """Extract side-effect import: import './file'"""
    import_path = match.group(1)
    
    if not import_path.startswith('.') and not import_path.startswith('/'):
        return None
    
    source_file = _resolve_js_path(import_path, current_dir)
    
    return ImportInfo(
        source_file=source_file,
        raw_statement=match.group(0),
        imported_items=[],
        import_type='side-effect'
    )


def _extract_js_export(match, current_dir, current_file, code_content):
    """Extract re-export: export { A } from './file'"""
    is_type_only = match.group(1) is not None
    export_clause = match.group(2)
    import_path = match.group(3)
    
    if not import_path.startswith('.') and not import_path.startswith('/'):
        return None
    
    source_file = _resolve_js_path(import_path, current_dir)
    
    imported_items = []
    import_type = 'named'
    alias = None
    
    if export_clause:
        if export_clause.startswith('{'):
            items_str = export_clause[1:-1] if export_clause.endswith('}') else export_clause[1:]
            for item in items_str.split(','):
                item = item.strip()
                if ' as ' in item:
                    imported_items.append(item.split(' as ')[0].strip())
                elif item:
                    imported_items.append(item)
        elif '*' in export_clause:
            import_type = 'namespace'
            imported_items = ['*']
            if ' as ' in export_clause:
                alias = export_clause.split(' as ')[1].strip()
    
    return ImportInfo(
        source_file=source_file,
        raw_statement=match.group(0),
        imported_items=imported_items,
        import_type=import_type,
        alias=alias,
        is_type_only=is_type_only
    )


def _resolve_js_path(import_path, current_dir):
    """Resolve JavaScript/TypeScript import path to file"""
    if import_path.startswith('/'):
        resolved = import_path.lstrip('/')
    else:
        resolved = os.path.join(current_dir, import_path)
        resolved = os.path.normpath(resolved)
    
    # Try different extensions
    extensions = ['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.json', '']
    for ext in extensions:
        path_with_ext = resolved + ext
        if os.path.exists(path_with_ext):
            return path_with_ext
    
    # Try as directory with index file
    for index_file in ['index.ts', 'index.tsx', 'index.js', 'index.jsx']:
        index_path = os.path.join(resolved, index_file)
        if os.path.exists(index_path):
            return index_path

    if '.' not in resolved:
        return resolved + '.ts'
    else:
        return resolved


def _extract_python_import(match, current_dir, current_file, code_content):
    """Extract Python 'from ... import ...' statement"""
    module_path = match.group(1)
    imports_str = match.group(2).strip()
    
    # Parse imported items
    imported_items = []
    for item in imports_str.split(','):
        item = item.strip()
        if ' as ' in item:
            imported_items.append(item.split(' as ')[0].strip())
        elif item and item != '*':
            imported_items.append(item)
        elif item == '*':
            imported_items = ['*']
            break
    
    # Resolve file path
    file_path = _resolve_python_path(module_path, current_dir)
    if not file_path:
        return None
    
    return ImportInfo(
        source_file=file_path,
        raw_statement=match.group(0),
        imported_items=imported_items,
        import_type='namespace' if '*' in imported_items else 'named'
    )


def _extract_python_import_simple(match, current_dir, current_file, code_content):
    """Extract Python 'import ...' statement"""
    imports_str = match.group(1)
    results = []
    
    for module in imports_str.split(','):
        module = module.strip()
        if not module:
            continue
        
        alias = None
        if ' as ' in module:
            module, alias = module.split(' as ')
            module = module.strip()
            alias = alias.strip()
        
        file_path = _resolve_python_path(module, current_dir)
        if file_path:
            results.append(ImportInfo(
                source_file=file_path,
                raw_statement=f"import {module}",
                imported_items=[module.split('.')[-1]],
                import_type='default',
                alias=alias
            ))
    
    return results if results else None


def _resolve_python_path(module_path, current_dir):
    """Resolve Python module to file path"""
    module_file_path = module_path.replace('.', os.sep)
    
    # Try as package
    package_init = os.path.normpath(os.path.join(current_dir, module_file_path, '__init__.py'))
    if os.path.exists(package_init):
        return package_init
    
    # Try as module
    module_file = os.path.normpath(os.path.join(current_dir, module_file_path + '.py'))
    if os.path.exists(module_file):
        return module_file
    
    return None


def _extract_c_include(match, current_dir, current_file, code_content):
    """Extract C/C++ #include statement"""
    include_path = match.group(1)
    resolved = os.path.normpath(os.path.join(current_dir, include_path))
    
    if not os.path.exists(resolved):
        return None
    
    return ImportInfo(
        source_file=resolved,
        raw_statement=match.group(0),
        imported_items=['*'],
        import_type='namespace'
    )


def _extract_rust_use(match, current_dir, current_file, code_content):
    """Extract Rust 'use' statement"""
    use_path = match.group(1)
    items_group = match.group(2)
    
    # Resolve file path
    module_path = use_path.replace('::', os.sep)
    resolved = os.path.normpath(os.path.join(current_dir, module_path + '.rs'))
    
    if not os.path.exists(resolved):
        return None
    
    # Parse imported items
    imported_items = []
    if items_group:
        # use module::{A, B, C}
        for item in items_group.split(','):
            imported_items.append(item.strip())
    else:
        # use module or use module::item
        imported_items = [use_path.split('::')[-1]]
    
    return ImportInfo(
        source_file=resolved,
        raw_statement=match.group(0),
        imported_items=imported_items,
        import_type='named'
    )


def _extract_java_import(match, current_dir, current_file, code_content):
    """Extract Java/Kotlin import statement"""
    is_static = match.group(1) is not None
    import_path = match.group(2)
    
    # Check if it's a wildcard import
    is_wildcard = import_path.endswith('.*')
    if is_wildcard:
        import_path = import_path[:-2]
    
    # Convert to file path
    parts = import_path.split('.')
    class_name = parts[-1]
    package_path = os.sep.join(parts[:-1]) if len(parts) > 1 else ''
    
    # Try .java and .kt
    for ext in ['.java', '.kt', '.kts']:
        if package_path:
            resolved = os.path.normpath(os.path.join(current_dir, package_path, class_name + ext))
        else:
            resolved = os.path.normpath(os.path.join(current_dir, class_name + ext))
        
        if os.path.exists(resolved):
            return ImportInfo(
                source_file=resolved,
                raw_statement=match.group(0),
                imported_items=['*'] if is_wildcard else [class_name],
                import_type='namespace' if is_wildcard else 'named'
            )
    
    return None


def _extract_go_import(match, current_dir, current_file, code_content):
    """Extract Go import statement"""
    alias = match.group(1)  # May be None
    import_path = match.group(2)
    
    # Skip external packages
    if '.' in import_path.split('/')[0]:
        return None
    
    package_name = import_path.split('/')[-1]
    
    return ImportInfo(
        source_file=import_path,
        raw_statement=match.group(0),
        imported_items=[package_name],
        import_type='default',
        alias=alias
    )


# ============================================================================
# Legacy Path Resolvers (for backward compatibility)
# ============================================================================

def main():
    """Test cases for parse_imports_detailed function"""
    print("=" * 80)
    print("Testing parse_imports_detailed function")
    print("=" * 80)
    
    # Test TypeScript imports with detailed information
    print("\n[Test 1] TypeScript Multi-line Imports")
    ts_code = '''import { v4 as uuidv4 } from 'uuid';
import * as bcrypt from 'bcryptjs';
import {
  User,
  UserRole,
  OAuthProvider,
  DEFAULT_USER_PREFERENCES
} from '../models/User';
import {
  Model,
  ModelCategory,
  Framework
} from '../models/Model';
import type { Dataset } from '../models/Dataset';
import './styles.css';
export { Button } from './components/Button';
'''
    result = parse_imports_detailed('backend/scripts/initData.ts', ts_code)
    print(f"Current file: backend/scripts/initData.ts")
    print(f"\nDetected {len(result)} imports:")
    for imp in result:
        print(f"\n  File: {imp.source_file}")
        print(f"    Type: {imp.import_type}")
        print(f"    Items: {imp.imported_items}")
        print(f"    Alias: {imp.alias}")
        print(f"    Type-only: {imp.is_type_only}")
        print(f"    Statement: {imp.raw_statement[:60]}...")
    
    # Test backward compatibility
    print("\n" + "=" * 80)
    print("[Test 2] Backward Compatibility Test")
    simple_result = parse_imports('backend/scripts/initData.ts', ts_code)
    print(f"parse_imports() returns list of file paths:")
    for path in simple_result:
        print(f"  - {path}")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()