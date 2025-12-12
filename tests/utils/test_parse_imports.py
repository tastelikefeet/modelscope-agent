"""
Unit tests for Import Parser
Tests Python, JavaScript/TypeScript, and Java import parsing
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from ms_agent.utils import parser_utils
parse_imports = parser_utils.parse_imports
ImportInfo = parser_utils.ImportInfo


class TestPythonImports(unittest.TestCase):
    """Test Python import parsing"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test.py')
        Path(self.test_file).touch()

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_from_import(self):
        """Test: from xx import xx"""
        content = "from typing import List, Dict"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertIn('List', imports[0].imported_items)
        self.assertIn('Dict', imports[0].imported_items)
        self.assertEqual(imports[0].import_type, 'named')

    def test_from_import_with_alias(self):
        """Test: from xx import xx as yy"""
        content = "from collections import defaultdict as dd"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertIn('defaultdict', imports[0].imported_items)

    def test_simple_import(self):
        """Test: import xx"""
        content = "import os"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].import_type, 'default')
        self.assertIn('os', imports[0].imported_items)

    def test_import_with_alias(self):
        """Test: import xx as yy"""
        content = "import numpy as np"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].alias, 'np')

    def test_import_star(self):
        """Test: from xx import *"""
        content = "from typing import *"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].import_type, 'namespace')
        self.assertIn('*', imports[0].imported_items)

    def test_multiline_import_parentheses(self):
        """Test multi-line import with parentheses"""
        content = '''
from typing import (
    List,
    Dict,
    Optional
)
'''
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertIn('List', imports[0].imported_items)
        self.assertIn('Dict', imports[0].imported_items)
        self.assertIn('Optional', imports[0].imported_items)

    def test_multiple_simple_imports(self):
        """Test: import xx, yy"""
        content = "import os, sys, json"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 3)
        items = [imp.imported_items[0] for imp in imports]
        self.assertIn('os', items)
        self.assertIn('sys', items)
        self.assertIn('json', items)

    def test_import_with_comment(self):
        """Test import with inline comment"""
        content = "from typing import List  # type hint"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertIn('List', imports[0].imported_items)
        # Comment should not be in imported items
        self.assertNotIn('#', str(imports[0].imported_items))


class TestJavaScriptImports(unittest.TestCase):
    """Test JavaScript/TypeScript import parsing"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test.ts')
        Path(self.test_file).touch()

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_named_import(self):
        """Test: import { xxx } from 'xxx'"""
        content = "import { useState, useEffect } from 'react'"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].import_type, 'named')
        self.assertIn('useState', imports[0].imported_items)
        self.assertIn('useEffect', imports[0].imported_items)

    def test_default_import(self):
        """Test: import xxx from 'xxx'"""
        content = "import React from 'react'"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].import_type, 'default')
        self.assertIn('React', imports[0].imported_items)

    def test_namespace_import(self):
        """Test: import * as xxx from 'xxx'"""
        content = "import * as utils from './utils'"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].import_type, 'namespace')
        self.assertEqual(imports[0].alias, 'utils')
        self.assertIn('*', imports[0].imported_items)

    def test_side_effect_import(self):
        """Test: import 'xx/xx'"""
        content = "import './styles.css'"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].import_type, 'side-effect')

    def test_type_import(self):
        """Test: import type { xxx } from 'xxx'"""
        content = "import type { User, Product } from './types'"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertTrue(imports[0].is_type_only)
        self.assertIn('User', imports[0].imported_items)
        self.assertIn('Product', imports[0].imported_items)

    def test_export_from(self):
        """Test: export { xxx } from 'xxx'"""
        content = "export { Button, Input } from './components'"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertIn('Button', imports[0].imported_items)
        self.assertIn('Input', imports[0].imported_items)

    def test_export_star(self):
        """Test: export * from 'xxx'"""
        content = "export * from './utils'"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].import_type, 'namespace')
        self.assertIn('*', imports[0].imported_items)

    def test_export_star_as(self):
        """Test: export * as name from 'xxx'"""
        content = "export * as helpers from './helpers'"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].alias, 'helpers')

    def test_multiline_named_import(self):
        """Test multiline named imports"""
        content = '''
import {
    Component1,
    Component2,
    Component3
} from './components'
'''
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertIn('Component1', imports[0].imported_items)
        self.assertIn('Component2', imports[0].imported_items)
        self.assertIn('Component3', imports[0].imported_items)

    def test_import_with_alias(self):
        """Test: import { xxx as yyy } from 'xxx'"""
        content = "import { useState as state } from 'react'"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        # Should extract original name before 'as'
        self.assertIn('useState', imports[0].imported_items)

    def test_css_module_import(self):
        """Test: import styles from './styles.module.css'"""
        content = "import styles from './styles.module.css'"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].import_type, 'default')
        self.assertIn('styles', imports[0].imported_items)


class TestJavaImports(unittest.TestCase):
    """Test Java import parsing"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'Test.java')
        Path(self.test_file).touch()

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_single_import(self):
        """Test: import package.Class"""
        content = "import java.util.List;"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertIn('List', imports[0].imported_items)

    def test_wildcard_import(self):
        """Test: import package.*"""
        content = "import java.util.*;"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].import_type, 'namespace')
        self.assertIn('*', imports[0].imported_items)

    def test_static_import(self):
        """Test: import static package.Class.method"""
        content = "import static java.lang.Math.PI;"
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        self.assertIn('PI', imports[0].imported_items)

    def test_multiple_imports(self):
        """Test multiple Java imports"""
        content = '''
import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
'''
        imports = parse_imports(self.test_file, content, self.temp_dir)

        self.assertEqual(len(imports), 3)
        items = [imp.imported_items[0] for imp in imports]
        self.assertIn('List', items)
        self.assertIn('ArrayList', items)
        self.assertIn('HashMap', items)


class TestPathResolution(unittest.TestCase):
    """Test path resolution functionality"""

    def setUp(self):
        """Set up test project structure"""
        self.temp_dir = tempfile.mkdtemp()

        # Create project structure
        src_dir = os.path.join(self.temp_dir, 'src')
        components_dir = os.path.join(src_dir, 'components')
        utils_dir = os.path.join(src_dir, 'utils')

        os.makedirs(components_dir)
        os.makedirs(utils_dir)

        # Create files
        Path(os.path.join(components_dir, 'Button.tsx')).touch()
        Path(os.path.join(components_dir, 'index.ts')).touch()
        Path(os.path.join(utils_dir, 'helpers.ts')).touch()
        self.app_file = os.path.join(src_dir, 'app.tsx')
        Path(self.app_file).touch()

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_relative_import_with_extension(self):
        """Test resolving relative imports with file extension"""
        content = "import { Button } from './components/Button.tsx'"
        imports = parse_imports(self.app_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        # Should resolve to actual file
        self.assertIn('Button', imports[0].source_file)

    def test_relative_import_without_extension(self):
        """Test resolving relative imports without extension"""
        content = "import { Button } from './components/Button'"
        imports = parse_imports(self.app_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        # Should auto-detect .tsx extension
        self.assertTrue(imports[0].source_file.endswith(('.tsx', 'Button')))

    def test_directory_import_resolves_to_index(self):
        """Test that directory imports resolve to index file"""
        content = "import { Component } from './components'"
        imports = parse_imports(self.app_file, content, self.temp_dir)

        self.assertEqual(len(imports), 1)
        # Should resolve to index.ts
        self.assertIn('index', imports[0].source_file)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_file(self):
        """Test parsing an empty file"""
        empty_file = os.path.join(self.temp_dir, 'empty.ts')
        Path(empty_file).touch()

        imports = parse_imports(empty_file, '', self.temp_dir)
        self.assertEqual(len(imports), 0)

    def test_file_with_no_imports(self):
        """Test parsing a file with no imports"""
        content = 'const x = 1;\nconst y = 2;\nconsole.log(x + y);'
        test_file = os.path.join(self.temp_dir, 'no_imports.ts')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)
        self.assertEqual(len(imports), 0)

    def test_malformed_import(self):
        """Test handling malformed import (should not crash)"""
        content = "import { from './broken'"
        test_file = os.path.join(self.temp_dir, 'broken.ts')
        Path(test_file).touch()

        # Should not crash
        imports = parse_imports(test_file, content, self.temp_dir)
        # May return empty or skip malformed import
        self.assertIsInstance(imports, list)

    def test_commented_imports_ignored(self):
        """Test that commented imports are not parsed"""
        content = '''
// import { useState } from 'react'
/* import { useEffect } from 'react' */
import { useCallback } from 'react'
'''
        test_file = os.path.join(self.temp_dir, 'commented.ts')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)

        # Only useCallback should be imported (comments should be ignored)
        all_items = []
        for imp in imports:
            all_items.extend(imp.imported_items)

        self.assertIn('useCallback', all_items)
        # Commented imports should NOT appear
        self.assertNotIn('useState', all_items)
        self.assertNotIn('useEffect', all_items)

    def test_complex_text_with_nested_patterns(self):
        """Test parsing complex text that previously caused nested quantifier issues"""
        # This is the actual content that caused catastrophic backtracking before
        content = '''现在我了解了AuthContext的导出内容。根据项目结构，这个index.js文件应该作为contexts目录的统一导出入口。让我编写这个文件：

<result>javascript: frontend/src/contexts/index.js

export { AuthProvider, useAuth } from './AuthContext';
export { ThemeProvider, useTheme } from './ThemeContext';
'''
        test_file = os.path.join(self.temp_dir, 'index.js')
        Path(test_file).touch()

        # Should not hang or crash, should parse the export statements
        imports = parse_imports(test_file, content, self.temp_dir)

        # Should find the two export statements
        self.assertGreaterEqual(len(imports), 2)
        
        # Verify it found the exports
        all_items = []
        for imp in imports:
            all_items.extend(imp.imported_items)
        
        self.assertIn('AuthProvider', all_items)
        self.assertIn('useAuth', all_items)
        self.assertIn('ThemeProvider', all_items)
        self.assertIn('useTheme', all_items)

    def test_mixed_import_export_statements(self):
        """Test file with mixed import and export statements"""
        content = '''
import React from 'react';
import { useState } from 'react';
export { Button } from './components';
export default App;
import './styles.css';
'''
        test_file = os.path.join(self.temp_dir, 'mixed.js')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)
        
        # Should find imports (not default export)
        self.assertGreaterEqual(len(imports), 4)
        all_items = []
        for imp in imports:
            all_items.extend(imp.imported_items)
        
        self.assertIn('React', all_items)
        self.assertIn('useState', all_items)
        self.assertIn('Button', all_items)

    def test_dynamic_import(self):
        """Test that dynamic imports are not parsed (only static imports)"""
        content = '''
import { useState } from 'react';
const LazyComponent = import('./LazyComponent');
const module = await import('./dynamic');
'''
        test_file = os.path.join(self.temp_dir, 'dynamic.js')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)
        
        # Should only find the static import
        self.assertEqual(len(imports), 1)
        self.assertIn('useState', imports[0].imported_items)

    def test_unicode_in_import_path(self):
        """Test imports with unicode characters in path"""
        content = '''
import { component } from './组件/模块';
import data from './données/fichier';
'''
        test_file = os.path.join(self.temp_dir, 'unicode.js')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)
        
        # Should parse successfully without crashing
        self.assertEqual(len(imports), 2)

    def test_very_long_import_list(self):
        """Test import with very long list of items"""
        items = ', '.join([f'Item{i}' for i in range(100)])
        content = f'import {{ {items} }} from "./large-module";'
        
        test_file = os.path.join(self.temp_dir, 'long.js')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)
        
        self.assertEqual(len(imports), 1)
        self.assertEqual(len(imports[0].imported_items), 100)
        self.assertIn('Item0', imports[0].imported_items)
        self.assertIn('Item99', imports[0].imported_items)

    def test_python_relative_imports(self):
        """Test Python relative imports with dots"""
        content = '''
from . import module1
from .. import module2
from ...package import module3
from .subpackage import Class1, Class2
'''
        test_file = os.path.join(self.temp_dir, 'relative.py')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)
        
        # Should parse all relative imports
        self.assertGreaterEqual(len(imports), 4)

    def test_python_future_imports(self):
        """Test Python __future__ imports"""
        content = '''
from __future__ import annotations
from __future__ import division, print_function
import sys
'''
        test_file = os.path.join(self.temp_dir, 'future.py')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)
        
        self.assertGreaterEqual(len(imports), 3)
        all_items = []
        for imp in imports:
            all_items.extend(imp.imported_items)
        
        self.assertIn('annotations', all_items)
        self.assertIn('division', all_items)

    def test_java_nested_class_import(self):
        """Test Java nested class imports"""
        content = '''
import java.util.Map.Entry;
import com.example.OuterClass.InnerClass;
'''
        test_file = os.path.join(self.temp_dir, 'Nested.java')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)
        
        self.assertEqual(len(imports), 2)
        items = [imp.imported_items[0] for imp in imports]
        self.assertIn('Entry', items)
        self.assertIn('InnerClass', items)

    def test_js_import_with_query_params(self):
        """Test JS imports with query parameters (e.g., Vite)"""
        content = '''
import Worker from './worker?worker';
import styles from './styles.css?inline';
'''
        test_file = os.path.join(self.temp_dir, 'query.js')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)
        
        # Should parse the imports (query params are part of path)
        self.assertEqual(len(imports), 2)

    def test_js_triple_slash_directives(self):
        """Test that TypeScript triple-slash directives are not confused with imports"""
        content = '''
/// <reference path="./types.d.ts" />
/// <reference types="node" />
import { Component } from 'react';
'''
        test_file = os.path.join(self.temp_dir, 'directives.ts')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)
        
        # Should only find the actual import, not triple-slash directives
        self.assertEqual(len(imports), 1)
        self.assertIn('Component', imports[0].imported_items)

    def test_multiline_import_with_comments(self):
        """Test multiline import with inline comments"""
        content = '''
import {
    Component1, // Main component
    Component2, /* Secondary */
    Component3
} from './components';
'''
        test_file = os.path.join(self.temp_dir, 'inline_comments.js')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)
        
        self.assertEqual(len(imports), 1)
        # Comments should not affect parsing
        self.assertEqual(len(imports[0].imported_items), 3)

    def test_python_import_with_parentheses_and_comments(self):
        """Test Python import with parentheses and comments"""
        content = '''
from typing import (
    List,  # For lists
    Dict,  # For dictionaries
    Optional,
)
'''
        test_file = os.path.join(self.temp_dir, 'py_comments.py')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)
        
        self.assertEqual(len(imports), 1)
        self.assertEqual(len(imports[0].imported_items), 3)

    def test_empty_braces_import(self):
        """Test import with empty braces"""
        content = "import { } from 'module';"
        test_file = os.path.join(self.temp_dir, 'empty.js')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)
        
        # Should handle gracefully
        self.assertIsInstance(imports, list)

    def test_consecutive_imports_no_newline(self):
        """Test consecutive imports without newlines (only first is matched by design)"""
        content = "import a from 'a';import b from 'b';import c from 'c';"
        test_file = os.path.join(self.temp_dir, 'consecutive.js')
        Path(test_file).touch()

        imports = parse_imports(test_file, content, self.temp_dir)
        
        # By design, regex with ^ only matches line start, so only first import is found
        # This is intentional to avoid false matches in strings
        self.assertGreaterEqual(len(imports), 1)
        self.assertIn('a', imports[0].imported_items)


if __name__ == '__main__':
    unittest.main()
