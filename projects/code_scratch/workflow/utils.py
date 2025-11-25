import os
import re
from typing import List


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


def parse_imports(current_file: str, code_content: str) -> List[str]:
    imports = []
    current_dir = os.path.dirname(current_file) if current_file else '.'

    # Import patterns for different languages
    patterns = [
        # Python: import/from ... import
        (r'^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.,\s]+))', ['py'], _resolve_python_import),

        # JavaScript/TypeScript: import/require
        (r'^\s*import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', ['js', 'ts', 'jsx', 'tsx', 'mjs', 'cjs'],
         _resolve_js_import),
        (r'^\s*import\s+[\'"]([^\'"]+)[\'"]', ['js', 'ts', 'jsx', 'tsx', 'mjs', 'cjs'], _resolve_js_import),
        (r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)', ['js', 'ts', 'jsx', 'tsx', 'mjs', 'cjs'], _resolve_js_import),
        (r'^\s*export\s+.*?from\s+[\'"]([^\'"]+)[\'"]', ['js', 'ts', 'jsx', 'tsx', 'mjs', 'cjs'],
         _resolve_js_import),

        # HTML: script src, link href, img src
        (r'<script[^>]+src=[\'"]([^\'"]+)[\'"]', ['html', 'htm'], _resolve_html_resource),
        (r'<link[^>]+href=[\'"]([^\'"]+)[\'"]', ['html', 'htm'], _resolve_html_resource),
        (r'<img[^>]+src=[\'"]([^\'"]+)[\'"]', ['html', 'htm'], _resolve_html_resource),
        (r'<iframe[^>]+src=[\'"]([^\'"]+)[\'"]', ['html', 'htm'], _resolve_html_resource),
        (r'<video[^>]+src=[\'"]([^\'"]+)[\'"]', ['html', 'htm'], _resolve_html_resource),
        (r'<audio[^>]+src=[\'"]([^\'"]+)[\'"]', ['html', 'htm'], _resolve_html_resource),
        (r'<source[^>]+src=[\'"]([^\'"]+)[\'"]', ['html', 'htm'], _resolve_html_resource),

        # C/C++: #include
        (r'^\s*#include\s+"([^"]+)"', ['c', 'cpp', 'cc', 'cxx', 'h', 'hpp'], _resolve_c_include),

        # Rust: use/mod
        (r'^\s*use\s+(?:crate::)?([\w:]+)', ['rs'], _resolve_rust_import),
        (r'^\s*mod\s+(\w+)', ['rs'], _resolve_rust_mod),

        # Java/Kotlin: import
        (r'^\s*import\s+([\w.]+)', ['java', 'kt', 'kts'], _resolve_java_import),

        # Go: import
        (r'^\s*import\s+"([^"]+)"', ['go'], _resolve_go_import),
        (r'^\s*import\s+\w+\s+"([^"]+)"', ['go'], _resolve_go_import),
    ]

    # Detect file extension
    file_ext = os.path.splitext(current_file)[1].lstrip('.').lower() if current_file else ''

    for line in code_content.split('\n'):
        for pattern, extensions, resolver in patterns:
            # Skip if file extension doesn't match
            if file_ext and file_ext not in extensions:
                continue

            # Use re.search for HTML patterns to match anywhere in the line
            # Use re.match for other patterns to match from line start
            if extensions in [['html', 'htm']]:
                match = re.search(pattern, line)
            else:
                match = re.match(pattern, line)
                
            if match:
                resolved = resolver(match, current_dir, current_file)
                if resolved:
                    if isinstance(resolved, list):
                        imports.extend(resolved)
                    else:
                        imports.append(resolved)

    # Remove duplicates while preserving order
    seen = set()
    unique_imports = []
    for imp in imports:
        if imp not in seen:
            seen.add(imp)
            unique_imports.append(imp)

    return unique_imports


def _resolve_python_import(match, current_dir, current_file):
    """Resolve Python import to file path"""
    imports = []
    from_module = match.group(1)
    import_modules = match.group(2)

    if from_module:
        # from xxx import yyy
        module_path = from_module.replace('.', os.sep)
        # Try as package
        package_init = os.path.normpath(os.path.join(current_dir, module_path, '__init__.py'))
        if _is_local_path(package_init):
            imports.append(package_init)
        # Try as module
        module_file = os.path.normpath(os.path.join(current_dir, module_path + '.py'))
        if _is_local_path(module_file):
            imports.append(module_file)

    if import_modules:
        # import xxx, yyy, zzz
        for module in import_modules.split(','):
            module = module.strip()
            if not module:
                continue
            module_path = module.replace('.', os.sep)
            # Try as package
            package_init = os.path.normpath(os.path.join(current_dir, module_path, '__init__.py'))
            if _is_local_path(package_init):
                imports.append(package_init)
            # Try as module
            module_file = os.path.normpath(os.path.join(current_dir, module_path + '.py'))
            if _is_local_path(module_file):
                imports.append(module_file)

    return imports


def _resolve_js_import(match, current_dir, current_file):
    """Resolve JavaScript/TypeScript import to file path"""
    import_path = match.group(1)

    # Skip external packages (don't start with ./ or ../)
    if not import_path.startswith('.') and not import_path.startswith('/'):
        return None

    # Resolve relative path
    resolved = os.path.normpath(os.path.join(current_dir, import_path))

    # Try different extensions
    extensions = ['', '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs', '.json']
    for ext in extensions:
        path_with_ext = resolved + ext
        if _is_local_path(path_with_ext):
            return path_with_ext

    # Try as directory with index file
    for index_file in ['index.js', 'index.ts', 'index.jsx', 'index.tsx']:
        index_path = os.path.join(resolved, index_file)
        if _is_local_path(index_path):
            return index_path

    return resolved if _is_local_path(resolved) else None


def _resolve_c_include(match, current_dir, current_file):
    """Resolve C/C++ include to file path"""
    include_path = match.group(1)
    resolved = os.path.normpath(os.path.join(current_dir, include_path))
    return resolved if _is_local_path(resolved) else None


def _resolve_rust_import(match, current_dir, current_file):
    """Resolve Rust use statement to file path"""
    use_path = match.group(1)
    # Convert :: to /
    module_path = use_path.replace('::', os.sep)
    resolved = os.path.normpath(os.path.join(current_dir, module_path + '.rs'))
    return resolved if _is_local_path(resolved) else None


def _resolve_rust_mod(match, current_dir, current_file):
    """Resolve Rust mod statement to file path"""
    mod_name = match.group(1)
    # Try mod_name.rs first
    resolved = os.path.normpath(os.path.join(current_dir, mod_name + '.rs'))
    if _is_local_path(resolved):
        return resolved
    # Try mod_name/mod.rs
    resolved = os.path.normpath(os.path.join(current_dir, mod_name, 'mod.rs'))
    return resolved if _is_local_path(resolved) else None


def _resolve_java_import(match, current_dir, current_file):
    """Resolve Java/Kotlin import to file path"""
    import_path = match.group(1)
    # Convert package.Class to package/Class
    parts = import_path.split('.')
    if not parts:
        return None

    # The last part is the class name
    class_name = parts[-1]
    package_path = os.sep.join(parts[:-1]) if len(parts) > 1 else ''

    # Try .java and .kt extensions
    for ext in ['.java', '.kt', '.kts']:
        if package_path:
            resolved = os.path.normpath(os.path.join(current_dir, package_path, class_name + ext))
        else:
            resolved = os.path.normpath(os.path.join(current_dir, class_name + ext))
        if _is_local_path(resolved):
            return resolved

    return None


def _resolve_go_import(match, current_dir, current_file):
    """Resolve Go import to file path"""
    import_path = match.group(1)
    # Skip external packages (contain domain names)
    if '.' in import_path.split('/')[0]:
        return None
    # For local packages, resolve relative to project root
    resolved = os.path.normpath(import_path)
    return resolved if _is_local_path(resolved) else None


def _resolve_html_resource(match, current_dir, current_file):
    """Resolve HTML resource (script, link, img, etc.) to file path"""
    resource_path = match.group(1)

    # Skip external URLs (http://, https://, //)
    if resource_path.startswith('http://') or resource_path.startswith('https://') or resource_path.startswith('//'):
        return None

    # Skip data URIs
    if resource_path.startswith('data:'):
        return None

    # Skip absolute URLs or CDN links
    if resource_path.startswith('/'):
        # Absolute path from root - normalize it
        resolved = os.path.normpath(resource_path.lstrip('/'))
    else:
        # Relative path
        resolved = os.path.normpath(os.path.join(current_dir, resource_path))

    return resolved if _is_local_path(resolved) else None


def _is_local_path(path):
    """Check if path looks like a local file (basic heuristic)"""
    # Don't validate existence, just check if it's a reasonable local path
    # Exclude absolute paths outside the project
    if os.path.isabs(path):
        return False
    # Exclude paths with URL-like patterns
    if '://' in path or path.startswith('http'):
        return False
    return True


def main():
    """Test cases for parse_imports function"""
    print("=" * 80)
    print("Testing parse_imports function")
    print("=" * 80)
    
    # Test 1: Python imports
    print("\n[Test 1] Python imports")
    python_code = """import os
import sys
from utils import helper
from package.module import function
import numpy, pandas
"""
    result = parse_imports('src/main.py', python_code)
    print(f"Current file: src/main.py")
    print(f"Code:\n{python_code}")
    print(f"Detected imports: {result}")
    
    # Test 2: JavaScript imports
    print("\n[Test 2] JavaScript imports")
    js_code = """import React from 'react';
import { useState } from 'react';
import App from './App';
import utils from '../utils/helper';
const config = require('./config.json');
export { default } from './components/Button';
"""
    result = parse_imports('src/components/Header.js', js_code)
    print(f"Current file: src/components/Header.js")
    print(f"Code:\n{js_code}")
    print(f"Detected imports: {result}")
    
    # Test 3: TypeScript imports
    print("\n[Test 3] TypeScript imports")
    ts_code = """import type { User } from './types';
import * as utils from '../utils';
import './styles.css';
import api from '@/api/client';
"""
    result = parse_imports('src/pages/home.ts', ts_code)
    print(f"Current file: src/pages/home.ts")
    print(f"Code:\n{ts_code}")
    print(f"Detected imports: {result}")
    
    # Test 4: HTML resources
    print("\n[Test 4] HTML resources")
    html_code = """<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="./styles/main.css">
    <script src="../js/app.js"></script>
    <script src="https://cdn.example.com/lib.js"></script>
</head>
<body>
    <img src="/images/logo.png" alt="Logo">
    <img src="https://example.com/image.jpg" alt="External">
    <video src="./media/video.mp4"></video>
    <iframe src="./embed.html"></iframe>
</body>
</html>
"""
    result = parse_imports('public/index.html', html_code)
    print(f"Current file: public/index.html")
    print(f"Code:\n{html_code}")
    print(f"Detected imports: {result}")
    
    # Test 5: C++ includes
    print("\n[Test 5] C++ includes")
    cpp_code = """#include <iostream>
#include <vector>
#include "utils.h"
#include "../common/types.h"
#include "lib/helpers.hpp"
"""
    result = parse_imports('src/main.cpp', cpp_code)
    print(f"Current file: src/main.cpp")
    print(f"Code:\n{cpp_code}")
    print(f"Detected imports: {result}")
    
    # Test 6: Rust imports
    print("\n[Test 6] Rust imports")
    rust_code = """use std::collections::HashMap;
use crate::utils::helper;
use super::common;
mod config;
mod database;
"""
    result = parse_imports('src/main.rs', rust_code)
    print(f"Current file: src/main.rs")
    print(f"Code:\n{rust_code}")
    print(f"Detected imports: {result}")
    
    # Test 7: Java imports
    print("\n[Test 7] Java imports")
    java_code = """package com.example.app;

import java.util.List;
import java.util.ArrayList;
import com.example.utils.Helper;
import com.example.models.User;
"""
    result = parse_imports('src/com/example/app/Main.java', java_code)
    print(f"Current file: src/com/example/app/Main.java")
    print(f"Code:\n{java_code}")
    print(f"Detected imports: {result}")
    
    # Test 8: Kotlin imports
    print("\n[Test 8] Kotlin imports")
    kotlin_code = """package com.example.app

import kotlin.collections.List
import com.example.utils.Helper
import com.example.models.User
"""
    result = parse_imports('src/com/example/app/Main.kt', kotlin_code)
    print(f"Current file: src/com/example/app/Main.kt")
    print(f"Code:\n{kotlin_code}")
    print(f"Detected imports: {result}")
    
    # Test 9: Go imports
    print("\n[Test 9] Go imports")
    go_code = """package main

import (
    "fmt"
    "os"
    "github.com/example/package"
    "myproject/utils"
    helper "myproject/internal/helper"
)
"""
    result = parse_imports('cmd/main.go', go_code)
    print(f"Current file: cmd/main.go")
    print(f"Code:\n{go_code}")
    print(f"Detected imports: {result}")
    
    # Test 10: Mixed content with edge cases
    print("\n[Test 10] Edge cases - Data URIs and external URLs")
    html_edge_cases = """<img src="data:image/png;base64,iVBORw0KG...">
<script src="https://cdnjs.cloudflare.com/lib.js"></script>
<link href="//fonts.googleapis.com/css">
<script src="./local.js"></script>
"""
    result = parse_imports('index.html', html_edge_cases)
    print(f"Current file: index.html")
    print(f"Code:\n{html_edge_cases}")
    print(f"Detected imports (should only include local.js): {result}")
    
    # Test 11: Nested directories
    print("\n[Test 11] Complex relative paths")
    complex_code = """from ...utils.parser import Parser
from ..models import User
from .helpers import validate
import config
"""
    result = parse_imports('src/app/services/auth.py', complex_code)
    print(f"Current file: src/app/services/auth.py")
    print(f"Code:\n{complex_code}")
    print(f"Detected imports: {result}")
    
    # Test 12: No file extension (generic test)
    print("\n[Test 12] No file extension")
    generic_code = """import React from './App';
from utils import helper
use crate::common;
#include "utils.h"
"""
    result = parse_imports('', generic_code)
    print(f"Current file: (empty string)")
    print(f"Code:\n{generic_code}")
    print(f"Detected imports (all patterns should match): {result}")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()