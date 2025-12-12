import os
import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


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
        items_str = ', '.join(
            self.imported_items) if self.imported_items else 'all'
        alias_str = f' as {self.alias}' if self.alias else ''
        return f"ImportInfo(file='{self.source_file}', items=[{items_str}]{alias_str})"


class BaseImportParser(ABC):
    """Base class for language-specific import parsers"""

    def __init__(self, output_dir: str, current_file: str, current_dir: str):
        self.output_dir = output_dir
        self.current_file = current_file
        self.current_dir = current_dir

    @abstractmethod
    def get_file_extensions(self) -> List[str]:
        """Return list of file extensions this parser handles"""
        pass

    @abstractmethod
    def parse(self, code_content: str) -> List[ImportInfo]:
        """Parse imports from code content"""
        pass

    def _resolve_path(self, module_path: str) -> Optional[str]:
        """Resolve module path to file path (to be overridden by subclasses)"""
        return None


class PythonImportParser(BaseImportParser):
    """Parser for Python import statements"""

    def get_file_extensions(self) -> List[str]:
        return ['py']

    def parse(self, code_content: str) -> List[ImportInfo]:
        imports = []

        # Pattern 1: from ... import ...
        from_pattern = r'^\s*from\s+([\w.]+)\s+import\s+(?:\(([^)]+)\)|([^\n]+))'
        for match in re.finditer(from_pattern, code_content, re.MULTILINE | re.DOTALL):
            info = self._extract_from_import(match, code_content)
            if info:
                imports.append(info)

        # Pattern 2: import ...
        import_pattern = r'^\s*import\s+([\w.,\s]+)'
        for match in re.finditer(import_pattern, code_content, re.MULTILINE):
            infos = self._extract_simple_import(match)
            imports.extend(infos)

        return imports

    def _extract_from_import(self, match, code_content) -> Optional[ImportInfo]:
        """Extract 'from ... import ...' statement"""
        module_path = match.group(1)
        # Group 2 is parenthesized multi-line imports, group 3 is single-line imports
        imports_str = (match.group(2) or match.group(3)).strip()

        # Remove inline comments
        lines = imports_str.split('\n')
        cleaned_items = []
        for line in lines:
            if '#' in line:
                line = line[:line.index('#')]
            cleaned_items.append(line.strip())
        imports_str = ','.join(cleaned_items)

        # Parse imported items
        imported_items = []
        for item in imports_str.split(','):
            item = item.strip()
            if not item:
                continue
            if ' as ' in item:
                imported_items.append(item.split(' as ')[0].strip())
            elif item != '*':
                imported_items.append(item)
            elif item == '*':
                imported_items = ['*']
                break

        # Resolve file path
        file_path = self._resolve_python_path(module_path)
        # If file not found, use module_path as source_file (could be stdlib or external package)
        if not file_path:
            file_path = module_path

        return ImportInfo(
            source_file=file_path,
            raw_statement=match.group(0),
            imported_items=imported_items,
            import_type='namespace' if '*' in imported_items else 'named'
        )

    def _extract_simple_import(self, match) -> List[ImportInfo]:
        """Extract 'import ...' statement"""
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

            file_path = self._resolve_python_path(module)
            # If not found, use module name (could be stdlib)
            if not file_path:
                file_path = module
            
            results.append(
                ImportInfo(
                    source_file=file_path,
                    raw_statement=f'import {module}',
                    imported_items=[module.split('.')[-1]],
                    import_type='default',
                    alias=alias
                )
            )

        return results

    def _resolve_python_path(self, module_path: str) -> Optional[str]:
        """Resolve Python module to file path"""
        module_file_path = module_path.replace('.', os.sep)

        # Try as package
        package_init = os.path.normpath(
            os.path.join(self.current_dir, module_file_path, '__init__.py'))
        if os.path.exists(package_init):
            return package_init

        # Try as module
        module_file = os.path.normpath(
            os.path.join(self.current_dir, module_file_path + '.py'))
        if os.path.exists(module_file):
            return module_file

        # Try from output_dir (absolute import)
        if self.output_dir:
            package_init_abs = os.path.normpath(
                os.path.join(self.output_dir, module_file_path, '__init__.py'))
            if os.path.exists(package_init_abs):
                return os.path.join(module_file_path, '__init__.py')

            module_file_abs = os.path.normpath(
                os.path.join(self.output_dir, module_file_path + '.py'))
            if os.path.exists(module_file_abs):
                return module_file_path + '.py'

        return None


class JavaScriptImportParser(BaseImportParser):
    """Parser for JavaScript/TypeScript import statements"""

    def __init__(self, output_dir: str, current_file: str, current_dir: str):
        super().__init__(output_dir, current_file, current_dir)
        self.path_aliases = self._load_path_aliases()

    def get_file_extensions(self) -> List[str]:
        return ['js', 'ts', 'jsx', 'tsx', 'mjs', 'cjs']

    def parse(self, code_content: str) -> List[ImportInfo]:
        imports = []

        # Pattern 1: Named import - import { A, B } from 'path' (supports multiline)
        named_pattern = r"^\s*import\s+(type\s+)?\{([^}]+)\}\s*from\s+['\"]([^'\"]+)['\"]"
        for match in re.finditer(named_pattern, code_content, re.MULTILINE | re.DOTALL):
            info = self._extract_named_import(match)
            if info:
                imports.append(info)

        # Pattern 2: Default import - import React from 'path'
        default_pattern = r"^\s*import\s+(type\s+)?(\w+)\s+from\s+['\"]([^'\"]+)['\"]"
        for match in re.finditer(default_pattern, code_content, re.MULTILINE):
            info = self._extract_default_import(match)
            if info:
                imports.append(info)

        # Pattern 3: Namespace import - import * as name from 'path'
        namespace_pattern = r"^\s*import\s+(type\s+)?\*\s+as\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]"
        for match in re.finditer(namespace_pattern, code_content, re.MULTILINE):
            info = self._extract_namespace_import(match)
            if info:
                imports.append(info)

        # Pattern 4: Side-effect import - import 'path'
        side_effect_pattern = r"^\s*import\s+['\"]([^'\"]+)['\"]"
        for match in re.finditer(side_effect_pattern, code_content, re.MULTILINE):
            info = self._extract_side_effect_import(match)
            if info:
                imports.append(info)

        # Pattern 5: Named re-export - export { A, B } from 'path' (supports multiline)
        export_named_pattern = r"^\s*export\s+(type\s+)?\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]"
        for match in re.finditer(export_named_pattern, code_content, re.MULTILINE | re.DOTALL):
            info = self._extract_export_named(match)
            if info:
                imports.append(info)

        # Pattern 6: Wildcard re-export - export * from 'path'
        export_wildcard_pattern = r"^\s*export\s+(type\s+)?\*\s+from\s+['\"]([^'\"]+)['\"]"
        for match in re.finditer(export_wildcard_pattern, code_content, re.MULTILINE):
            info = self._extract_export_wildcard(match)
            if info:
                imports.append(info)

        # Pattern 7: Named wildcard re-export - export * as name from 'path'
        export_named_wildcard_pattern = r"^\s*export\s+(type\s+)?\*\s+as\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]"
        for match in re.finditer(export_named_wildcard_pattern, code_content, re.MULTILINE):
            info = self._extract_export_named_wildcard(match)
            if info:
                imports.append(info)

        return imports

    def _extract_named_import(self, match) -> Optional[ImportInfo]:
        """Extract: import { A, B } from 'path'"""
        is_type = bool(match.group(1))
        items_str = match.group(2).strip()
        import_path = match.group(3)

        items = [item.split(' as ')[0].strip() for item in items_str.split(',') if item.strip()]
        resolved_path = self._resolve_js_path(import_path)
        # If not resolved, use import_path as-is (external package)
        if not resolved_path:
            resolved_path = import_path

        return ImportInfo(
            source_file=resolved_path,
            raw_statement=match.group(0),
            imported_items=items,
            import_type='named',
            is_type_only=is_type
        )

    def _extract_default_import(self, match) -> Optional[ImportInfo]:
        """Extract: import React from 'path'"""
        is_type = bool(match.group(1))
        name = match.group(2)
        import_path = match.group(3)

        resolved_path = self._resolve_js_path(import_path)
        # If not resolved, use import_path as-is (external package)
        if not resolved_path:
            resolved_path = import_path

        return ImportInfo(
            source_file=resolved_path,
            raw_statement=match.group(0),
            imported_items=[name],
            import_type='default',
            is_type_only=is_type
        )

    def _extract_namespace_import(self, match) -> Optional[ImportInfo]:
        """Extract: import * as name from 'path'"""
        is_type = bool(match.group(1))
        name = match.group(2)
        import_path = match.group(3)

        resolved_path = self._resolve_js_path(import_path)
        # If not resolved, use import_path as-is (external package)
        if not resolved_path:
            resolved_path = import_path

        return ImportInfo(
            source_file=resolved_path,
            raw_statement=match.group(0),
            imported_items=['*'],
            import_type='namespace',
            alias=name,
            is_type_only=is_type
        )

    def _extract_side_effect_import(self, match) -> Optional[ImportInfo]:
        """Extract: import 'path'"""
        import_path = match.group(1)
        resolved_path = self._resolve_js_path(import_path)
        # If not resolved, use import_path as-is (external package)
        if not resolved_path:
            resolved_path = import_path

        return ImportInfo(
            source_file=resolved_path,
            raw_statement=match.group(0),
            imported_items=[],
            import_type='side-effect'
        )

    def _extract_export_named(self, match) -> Optional[ImportInfo]:
        """Extract: export { A, B } from 'path'"""
        is_type = bool(match.group(1))
        items_str = match.group(2).strip()
        import_path = match.group(3)

        items = [item.split(' as ')[0].strip() for item in items_str.split(',') if item.strip()]
        resolved_path = self._resolve_js_path(import_path)
        # If not resolved, use import_path as-is (external package)
        if not resolved_path:
            resolved_path = import_path

        return ImportInfo(
            source_file=resolved_path,
            raw_statement=match.group(0),
            imported_items=items,
            import_type='named',
            is_type_only=is_type
        )

    def _extract_export_wildcard(self, match) -> Optional[ImportInfo]:
        """Extract: export * from 'path'"""
        is_type = bool(match.group(1))
        import_path = match.group(2)

        resolved_path = self._resolve_js_path(import_path)
        # If not resolved, use import_path as-is (external package)
        if not resolved_path:
            resolved_path = import_path

        return ImportInfo(
            source_file=resolved_path,
            raw_statement=match.group(0),
            imported_items=['*'],
            import_type='namespace',
            is_type_only=is_type
        )

    def _extract_export_named_wildcard(self, match) -> Optional[ImportInfo]:
        """Extract: export * as name from 'path'"""
        is_type = bool(match.group(1))
        name = match.group(2)
        import_path = match.group(3)

        resolved_path = self._resolve_js_path(import_path)
        # If not resolved, use import_path as-is (external package)
        if not resolved_path:
            resolved_path = import_path

        return ImportInfo(
            source_file=resolved_path,
            raw_statement=match.group(0),
            imported_items=['*'],
            import_type='namespace',
            alias=name,
            is_type_only=is_type
        )

    def _resolve_js_path(self, import_path: str) -> Optional[str]:
        """Resolve JavaScript/TypeScript import path to file"""
        # Check for path alias
        resolved = self._resolve_alias_path(import_path)
        if resolved:
            import_path = resolved

        # Handle absolute paths
        if import_path.startswith('/'):
            resolved = import_path.lstrip('/')
        else:
            resolved = os.path.join(self.current_dir, import_path)
            resolved = os.path.normpath(resolved)

        # Try as directory with index file first
        if os.path.isdir(resolved):
            for index_file in ['index.ts', 'index.tsx', 'index.js', 'index.jsx']:
                index_path = os.path.join(resolved, index_file)
                if os.path.exists(index_path):
                    return index_path

        # Try different extensions
        extensions = [
            '.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.json',
            '.css', '.scss', '.sass', '.less', '.module.css', '.module.scss',
        ]

        for ext in extensions:
            path_with_ext = resolved + ext
            if os.path.exists(path_with_ext):
                return path_with_ext

        # If path already has an extension or is a known module, return it
        if os.path.exists(resolved):
            return resolved

        return resolved if '.' in resolved[1:] else None

    def _load_path_aliases(self) -> Dict[str, str]:
        """Load path aliases from tsconfig.json and vite.config"""
        aliases = {}
        excluded_dirs = {'node_modules', 'dist', 'build', '.git', '__pycache__'}

        # Search for config files
        for root, dirs, files in os.walk(self.output_dir):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            # tsconfig.json
            if 'tsconfig.json' in files:
                self._parse_tsconfig_aliases(os.path.join(root, 'tsconfig.json'), root, aliases)

            # vite.config.*
            for config_file in ['vite.config.js', 'vite.config.ts', 'vite.config.mjs']:
                if config_file in files:
                    self._parse_vite_config_aliases(os.path.join(root, config_file), root, aliases)

        # Default aliases
        if not aliases:
            for root, dirs, files in os.walk(self.output_dir):
                dirs[:] = [d for d in dirs if d not in excluded_dirs]
                if 'src' in dirs:
                    aliases['@'] = os.path.join(root, 'src')
                    aliases['~'] = root
                    break

        return aliases

    def _parse_tsconfig_aliases(self, tsconfig_path: str, base_dir: str, aliases: Dict[str, str]):
        """Parse tsconfig.json and extract path aliases"""
        try:
            with open(tsconfig_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Remove comments
                content = re.sub(r'//.*?\n|/\*.*?\*/', '', content, flags=re.DOTALL)
                tsconfig = json.loads(content)

                if 'compilerOptions' in tsconfig and 'paths' in tsconfig['compilerOptions']:
                    base_url = tsconfig['compilerOptions'].get('baseUrl', '.')
                    for alias, paths in tsconfig['compilerOptions']['paths'].items():
                        clean_alias = alias.rstrip('/*')
                        if paths and len(paths) > 0:
                            target = paths[0].rstrip('/*')
                            resolved_target = os.path.normpath(os.path.join(base_dir, base_url, target))
                            if clean_alias not in aliases:
                                aliases[clean_alias] = resolved_target
        except (json.JSONDecodeError, IOError, KeyError):
            pass

    def _parse_vite_config_aliases(self, config_path: str, base_dir: str, aliases: Dict[str, str]):
        """Parse vite.config and extract path aliases"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                alias_pattern = r"['\"]([^'\"]+)['\"]\s*:\s*(?:path\.resolve\([^,]+,\s*['\"]([^'\"]+)['\"]\)|['\"]([^'\"]+)['\"])"
                for match in re.finditer(alias_pattern, content):
                    alias_key = match.group(1)
                    target = match.group(2) or match.group(3)
                    if target:
                        target = target.lstrip('/')
                        resolved_target = os.path.join(base_dir, target)
                        if alias_key not in aliases:
                            aliases[alias_key] = resolved_target
        except IOError:
            pass

    def _resolve_alias_path(self, import_path: str) -> Optional[str]:
        """Resolve path alias to actual path"""
        for alias, target in self.path_aliases.items():
            if import_path == alias:
                return target
            elif import_path.startswith(alias + '/'):
                remainder = import_path[len(alias) + 1:]
                return os.path.join(target, remainder)
        return None


class JavaImportParser(BaseImportParser):
    """Parser for Java import statements"""

    def get_file_extensions(self) -> List[str]:
        return ['java']

    def parse(self, code_content: str) -> List[ImportInfo]:
        imports = []

        # Pattern: import [static] package.Class[.*]; or import [static] package.*;
        import_pattern = r'^\s*import\s+(static\s+)?((?:[\w]+\.)*[\w*]+);?'
        for match in re.finditer(import_pattern, code_content, re.MULTILINE):
            info = self._extract_java_import(match)
            if info:
                imports.append(info)

        return imports

    def _extract_java_import(self, match) -> Optional[ImportInfo]:
        """Extract Java import statement"""
        is_static = bool(match.group(1))
        import_path = match.group(2)

        # Resolve to file path
        file_path = self._resolve_java_path(import_path)
        # If not resolved, use import_path as-is (stdlib or external package)
        if not file_path:
            file_path = import_path

        # Determine import type
        if import_path.endswith('.*'):
            import_type = 'namespace'
            items = ['*']
        else:
            import_type = 'named'
            items = [import_path.split('.')[-1]]

        return ImportInfo(
            source_file=file_path,
            raw_statement=match.group(0),
            imported_items=items,
            import_type=import_type
        )

    def _resolve_java_path(self, import_path: str) -> Optional[str]:
        """Resolve Java import to file path"""
        # Remove .* if present
        if import_path.endswith('.*'):
            import_path = import_path[:-2]

        # Convert package.Class to path/Class.java
        file_path = import_path.replace('.', os.sep) + '.java'
        full_path = os.path.join(self.output_dir, file_path)

        if os.path.exists(full_path):
            return file_path

        return None


class ImportParserFactory:
    """Factory to get appropriate parser for file type"""

    @staticmethod
    def get_parser(file_ext: str, output_dir: str, current_file: str, current_dir: str) -> Optional[BaseImportParser]:
        """Get parser instance for given file extension"""
        parsers = [
            PythonImportParser,
            JavaScriptImportParser,
            JavaImportParser,
        ]

        for parser_class in parsers:
            parser = parser_class(output_dir, current_file, current_dir)
            if file_ext in parser.get_file_extensions():
                return parser

        return None


def parse_imports(current_file: str, code_content: str, output_dir: str) -> List[ImportInfo]:
    """
    Parse imports from code content (main entry point for backward compatibility)

    Args:
        current_file: Path to the file being parsed
        code_content: Content of the file
        output_dir: Root directory of the project

    Returns:
        List of ImportInfo objects
    """
    # Detect file extension
    file_ext = os.path.splitext(current_file)[1].lstrip('.').lower() if current_file else ''
    current_dir = os.path.dirname(current_file) if current_file else '.'

    # Get appropriate parser
    parser = ImportParserFactory.get_parser(file_ext, output_dir, current_file, current_dir)
    if not parser:
        return []

    # Parse imports
    return parser.parse(code_content)
