import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ms_agent.tools.base import ToolBase
from ms_agent.utils import get_logger
from ms_agent.utils.constants import DEFAULT_OUTPUT_DIR, DEFAULT_INDEX_DIR, DEFAULT_LOCK_DIR

logger = get_logger()


class LSPServer:
    """Base class for LSP server management"""

    def __init__(self, config):
        self.config = config
        self.process = None
        self.stdin = None
        self.stdout = None
        self.message_id = 0
        self.initialized = False
        self.output_dir = getattr(self.config, 'output_dir',
                                  DEFAULT_OUTPUT_DIR)
        self.workspace_dir = Path(self.output_dir).resolve()
        self.index_dir = os.path.join(self.output_dir, DEFAULT_INDEX_DIR)
        self.lock_dir = os.path.join(self.output_dir, DEFAULT_LOCK_DIR)
        self.diagnostics_cache: Dict[str, List[dict]] = {}
        
    async def start(self) -> bool:
        """Start the LSP server process"""
        raise NotImplementedError
        
    async def stop(self):
        """Stop the LSP server process"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                logger.error(f"Error stopping LSP server: {e}")
                
    async def send_request(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request to the LSP server"""
        if not self.process or not self.stdin or not self.stdout:
            raise RuntimeError("LSP server not started")
            
        self.message_id += 1
        request_id = self.message_id
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        
        content = json.dumps(request)
        message = f"Content-Length: {len(content)}\r\n\r\n{content}"
        
        try:
            self.stdin.write(message.encode('utf-8'))
            await self.stdin.drain()

            max_retries = 20
            for _ in range(max_retries):
                msg = await self._read_message()
                
                # Check if it's the response we're waiting for
                if "id" in msg and msg["id"] == request_id:
                    return msg
                
                # It's a notification (no id) or response for different request
                # Log and continue reading
                if "method" in msg:
                    logger.debug(f"Received notification during request: {msg.get('method')}")
                    continue
                    
            logger.warning(f"No response received for request {request_id} after {max_retries} attempts")
            return {"error": "No response received"}
            
        except Exception as e:
            logger.error(f"Error sending LSP request: {e}")
            return {"error": str(e)}
            
    async def send_notification(self, method: str, params: dict = None):
        """Send a JSON-RPC notification to the LSP server"""
        if not self.process or not self.stdin:
            raise RuntimeError("LSP server not started")
            
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        
        content = json.dumps(notification)
        message = f"Content-Length: {len(content)}\r\n\r\n{content}"
        
        try:
            self.stdin.write(message.encode('utf-8'))
            await self.stdin.drain()
        except Exception as e:
            logger.error(f"Error sending LSP notification: {e}")
            
    async def _read_message(self) -> dict:
        """Read a JSON-RPC message from the LSP server"""
        # Read headers
        headers = {}
        while True:
            line = await self.stdout.readline()
            line = line.decode('utf-8').strip()
            if not line:
                break
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip()] = value.strip()
                
        # Read content
        content_length = int(headers.get('Content-Length', 0))
        if content_length > 0:
            content = await self.stdout.readexactly(content_length)
            return json.loads(content.decode('utf-8'))
        return {}
        
    async def initialize(self):
        """Initialize the LSP server and wait for it to be ready"""
        response = await self.send_request("initialize", {
            "processId": os.getpid(),
            "rootUri": self.workspace_dir.as_uri(),
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {},
                    "synchronization": {
                        "didOpen": True,
                        "didChange": True,
                        "didClose": True
                    }
                }
            }
        })
        
        if "result" in response:
            await self.send_notification("initialized", {})
            
            # CRITICAL: Wait for server to be fully ready
            # Read and discard any startup messages
            await asyncio.sleep(1.0)  # Give server time to complete initialization
            
            # Consume any pending messages (like "starting" notifications)
            try:
                for _ in range(10):
                    try:
                        await asyncio.wait_for(self._read_message(), timeout=2.0)
                    except asyncio.TimeoutError:
                        break
            except Exception as e:
                logger.debug(f"Cleared startup messages: {e}")
            
            self.initialized = True
            logger.info("LSP server fully initialized and ready")
            return True
        
        logger.error(f"LSP initialization failed: {response}")
        return False
        
    async def open_document(self, file_path: str, content: str, language_id: str):
        """Open a document in the LSP server"""
        file_uri = Path(file_path).resolve().as_uri()
        await self.send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": file_uri,
                "languageId": language_id,
                "version": 1,
                "text": content
            }
        })
    
    async def close_document(self, file_path: str):
        """Close a document to clean up old index"""
        file_uri = Path(file_path).resolve().as_uri()
        await self.send_notification("textDocument/didClose", {
            "textDocument": {
                "uri": file_uri
            }
        })
        
    async def update_document(self, file_path: str, content: str, version: int = 2):
        """Update a document in the LSP server"""
        file_uri = Path(file_path).resolve().as_uri()
        await self.send_notification("textDocument/didChange", {
            "textDocument": {
                "uri": file_uri,
                "version": version
            },
            "contentChanges": [{"text": content}]
        })
        
    async def get_diagnostics(self, file_path: str, wait_time: float = 0.5, use_cache: bool = True) -> List[dict]:
        """Get diagnostics for a file
        
        Args:
            file_path: Path to the file
            wait_time: Time to wait for diagnostics (longer for new files)
            use_cache: Whether to use cached diagnostics on timeout
        """
        # Wait a bit for diagnostics to be computed
        await asyncio.sleep(wait_time)
        
        file_uri = Path(file_path).resolve().as_uri()
        
        # Try to read pending diagnostics messages
        diagnostics = []
        found_target = False
        max_attempts = 20
        consecutive_timeouts = 0
        
        for _ in range(max_attempts):
            try:
                msg = await asyncio.wait_for(self._read_message(), timeout=2.0)
                consecutive_timeouts = 0
                
                if msg.get("method") == "textDocument/publishDiagnostics":
                    current_uri = msg.get("params", {}).get("uri")
                    current_diags = msg.get("params", {}).get("diagnostics", [])
                    
                    # Update cache for all received diagnostics
                    self.diagnostics_cache[current_uri] = current_diags
                    logger.debug(f'Cached diagnostics for {current_uri}')
                    
                    # Check if this is the target file
                    if current_uri == file_uri:
                        diagnostics = current_diags
                        found_target = True
                        logger.debug(f'Found target diagnostics for {file_uri}')
                        break
                        
            except asyncio.TimeoutError:
                consecutive_timeouts += 1
                # After 3 consecutive timeouts, stop waiting
                if consecutive_timeouts >= 3:
                    logger.debug(f'Stopped after {consecutive_timeouts} consecutive timeouts')
                    break
                continue
        
        # If not found, try to use cache
        if not found_target:
            if use_cache and file_uri in self.diagnostics_cache:
                diagnostics = self.diagnostics_cache[file_uri]
                logger.debug(f'Using cached diagnostics for {file_uri}')
            else:
                logger.warning(f'No diagnostics found for {file_uri} (cache available: {file_uri in self.diagnostics_cache})')
                # Return empty instead of raising error
                diagnostics = []
        
        # Filter out 'unused' hints
        diagnostics = [d for d in diagnostics if not isinstance(d.get('code'), str) or 'unused' not in d['code'].lower()]
        return diagnostics


class VolarLSPServer(LSPServer):
    """Vue Language Server (Volar)"""
    
    async def start(self) -> bool:
        """Start Volar language server"""
        try:
            # Check if @vue/language-server is installed
            check_process = await asyncio.create_subprocess_exec(
                "npx", "@vue/language-server", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await check_process.communicate()
            
            if check_process.returncode != 0:
                logger.warning("Volar not found. Install with: npm install -g @vue/language-server")
                return False
                
            # Start vue-language-server
            self.process = await asyncio.create_subprocess_exec(
                "npx", "@vue/language-server", "--stdio",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_dir)
            )
            
            self.stdin = self.process.stdin
            self.stdout = self.process.stdout
            
            # Initialize the server
            return await self.initialize()
            
        except FileNotFoundError:
            logger.error("vue-language-server not found. Install with: npm install -g @vue/language-server")
            return False
        except Exception as e:
            logger.error(f"Failed to start Volar LSP server: {e}")
            return False


class TypeScriptLSPServer(LSPServer):
    """TypeScript/JavaScript LSP server (tsserver)"""
    
    async def start(self) -> bool:
        """Start tsserver"""
        try:
            # Check if typescript is installed
            check_process = await asyncio.create_subprocess_exec(
                "npx", "tsc", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await check_process.communicate()
            
            if check_process.returncode != 0:
                logger.warning("TypeScript not found. Install with: npm install -g typescript")
                return False
                
            # Start typescript-language-server
            self.process = await asyncio.create_subprocess_exec(
                "npx", "typescript-language-server", "--stdio",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_dir)
            )
            
            self.stdin = self.process.stdin
            self.stdout = self.process.stdout
            
            # Initialize the server
            return await self.initialize()
            
        except FileNotFoundError:
            logger.error("typescript-language-server not found. Install with: npm install -g typescript-language-server")
            return False
        except Exception as e:
            logger.error(f"Failed to start TypeScript LSP server: {e}")
            return False


class PythonLSPServer(LSPServer):
    """Python LSP server (pyright)"""
    
    async def start(self) -> bool:
        """Start pyright"""
        try:
            # Check if pyright is installed
            check_process = await asyncio.create_subprocess_exec(
                "pyright", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await check_process.communicate()
            
            if check_process.returncode != 0:
                logger.warning("Pyright not found. Install with: pip install pyright")
                return False
                
            # Start pyright langserver
            self.process = await asyncio.create_subprocess_exec(
                "pyright-langserver", "--stdio",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_dir)
            )
            
            self.stdin = self.process.stdin
            self.stdout = self.process.stdout
            
            # Initialize the server
            return await self.initialize()
            
        except FileNotFoundError:
            logger.error("pyright-langserver not found. Install with: pip install pyright")
            return False
        except Exception as e:
            logger.error(f"Failed to start Python LSP server: {e}")
            return False


class JavaLSPServer(LSPServer):
    """Java LSP server (Eclipse JDT Language Server)"""
    
    async def start(self) -> bool:
        """Start jdtls (Eclipse JDT Language Server)"""
        try:
            # Check if jdtls is available
            # Try common installation locations
            jdtls_paths = [
                "/usr/local/bin/jdtls",
                "/opt/homebrew/bin/jdtls",
                os.path.expanduser("~/.local/bin/jdtls"),
            ]
            
            jdtls_cmd = None
            for path in jdtls_paths:
                if os.path.exists(path):
                    jdtls_cmd = path
                    break
            
            if not jdtls_cmd:
                # Try to find in PATH
                check_process = await asyncio.create_subprocess_exec(
                    "which", "jdtls",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await check_process.communicate()
                if check_process.returncode == 0:
                    jdtls_cmd = stdout.decode('utf-8').strip()
            
            if not jdtls_cmd:
                logger.warning(
                    "jdtls not found. Install Eclipse JDT Language Server.\n"
                    "macOS: brew install jdtls\n"
                    "Or download from: https://download.eclipse.org/jdtls/snapshots/"
                )
                return False
            
            # Create workspace data directory for jdtls
            workspace_data_dir = Path(self.workspace_dir) / ".jdtls_workspace"
            workspace_data_dir.mkdir(exist_ok=True)
            
            # Start jdtls
            self.process = await asyncio.create_subprocess_exec(
                jdtls_cmd,
                "-data", str(workspace_data_dir),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_dir)
            )
            
            self.stdin = self.process.stdin
            self.stdout = self.process.stdout
            
            # Initialize the server
            return await self.initialize()
            
        except FileNotFoundError:
            logger.error(
                "jdtls not found. Install Eclipse JDT Language Server.\n"
                "macOS: brew install jdtls\n"
                "Or download from: https://download.eclipse.org/jdtls/snapshots/"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to start Java LSP server: {e}")
            return False


class LSPCodeServer(ToolBase):
    """LSP Code Server Tool for code quality checking
    
    Supports:
    - TypeScript/JavaScript (tsserver)
    - Python (pyright)
    - Java (jdtls)
    
    Features:
    1. Check entire directory
    2. Incremental code checking
    3. Detect issues in code segments
    """

    skip_files = [
        'vite.config.ts', 'vite.config.js',
        'webpack.config.js', 'webpack.config.ts',
        'rollup.config.js', 'rollup.config.ts',
        'next.config.js', 'next.config.ts',
        'tsconfig.json', 'jsconfig.json',
        'package.json', 'pom.xml', 'build.gradle'
    ]

    def __init__(self, config):
        super().__init__(config)
        self.servers: Dict[str, LSPServer] = {}
        self.file_versions: Dict[str, int] = {}
        self.output_dir = getattr(self.config, 'output_dir',
                                  DEFAULT_OUTPUT_DIR)
        self.workspace_dir = self.output_dir
        self.index_dir = os.path.join(self.output_dir, DEFAULT_INDEX_DIR)
        self.lock_dir = os.path.join(self.output_dir, DEFAULT_LOCK_DIR)
        
    async def connect(self) -> None:
        """Initialize LSP servers"""
        logger.info("LSP Code Server connecting...")
        
    async def cleanup(self) -> None:
        """Stop all LSP servers and clear indexes"""
        # Close all open documents first
        for file_path in list(self.file_versions.keys()):
            for lang, server in self.servers.items():
                try:
                    await server.close_document(file_path)
                except Exception as e:
                    logger.debug(f"Error closing document {file_path}: {e}")
        
        # Clear version tracking
        self.file_versions.clear()
        
        # Stop all servers
        for server in self.servers.values():
            await server.stop()
        self.servers.clear()
        logger.info("All LSP servers stopped and indexes cleared")
        
    async def _get_tools_inner(self) -> Dict[str, Any]:
        """Get available tools"""
        return {
            "lsp_code_server": [
                {
                    "tool_name": "check_directory",
                    "description": (
                        "Check all code files in a directory for errors and issues. "
                        "Supports TypeScript/JavaScript, Python, and Java files. "
                        "Returns a summary of all diagnostics found."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Path to the directory to check (relative to workspace)"
                            },
                            "language": {
                                "type": "string",
                                "enum": ["typescript", "python", "java", "vue"],
                                "description": "Programming language to check (typescript for JS/TS, vue for Vue projects, python for Python, java for Java)"
                            }
                        },
                        "required": ["directory", "language"]
                    }
                },
                {
                    "tool_name": "check_code_content",
                    "description": (
                        "Check a specific code segment for errors and issues. "
                        "Can be used to validate code before writing to file. "
                        "Returns detailed diagnostics including line numbers and error messages."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The code content to check"
                            },
                            "language": {
                                "type": "string",
                                "enum": ["typescript", "javascript", "python", "java", "vue"],
                                "description": "Programming language of the code"
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Optional file path for context (helps with import resolution)"
                            }
                        },
                        "required": ["content", "language"]
                    }
                },
                {
                    "tool_name": "update_and_check",
                    "description": (
                        "Incrementally update a file's content and check for errors. "
                        "Used during code generation to validate each N lines. "
                        "More efficient than checking from scratch each time."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file (relative to workspace)"
                            },
                            "content": {
                                "type": "string",
                                "description": "Updated file content"
                            },
                            "language": {
                                "type": "string",
                                "enum": ["typescript", "javascript", "python", "java", "vue"],
                                "description": "Programming language of the file"
                            }
                        },
                        "required": ["file_path", "content", "language"]
                    }
                }
            ]
        }
        
    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict) -> str:
        """Call a tool"""
        if tool_name == "check_directory":
            return await self._check_directory(
                tool_args["directory"],
                tool_args["language"]
            )
        elif tool_name == "check_code_content":
            return await self._check_code_content(
                tool_args["content"],
                tool_args["language"],
                tool_args.get("file_path")
            )
        elif tool_name == "update_and_check":
            return await self._update_and_check(
                tool_args["file_path"],
                tool_args["content"],
                tool_args["language"]
            )
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
            
    async def _get_or_create_server(self, language: str) -> Optional[LSPServer]:
        """Get or create an LSP server for the given language"""
        if language in self.servers:
            return self.servers[language]
            
        # Normalize language
        lang_key = language.lower()
        if lang_key in ["javascript", "typescript", "js", "ts"]:
            lang_key = "typescript"
        elif lang_key in ["python", "py"]:
            lang_key = "python"
        elif lang_key in ["java"]:
            lang_key = "java"
        elif lang_key in ["vue"]:
            lang_key = "vue"
        else:
            return None
            
        # Create server
        if lang_key == "typescript":
            server = TypeScriptLSPServer(self.config)
        elif lang_key == "python":
            server = PythonLSPServer(self.config)
        elif lang_key == "java":
            server = JavaLSPServer(self.config)
        elif lang_key == "vue":
            server = VolarLSPServer(self.config)
        else:
            return None
            
        # Start server
        if await server.start():
            self.servers[lang_key] = server
            return server
        return None
        
    async def _check_directory(self, directory: str, language: str) -> str:
        """Check all files in a directory"""
        try:
            server = await self._get_or_create_server(language)
            if not server:
                return json.dumps({
                    "error": f"Failed to start LSP server for {language}"
                })
                
            dir_path = Path(self.workspace_dir) / directory
            if not dir_path.exists() or not dir_path.is_dir():
                return json.dumps({
                    "error": f"Directory not found: {directory}"
                })
                
            # Determine file extensions
            if language.lower() in ["typescript", "javascript"]:
                extensions = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"]
                lang_id = "typescript"
            elif language.lower() in ["vue"]:
                extensions = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue"]
                lang_id = "vue"
            elif language.lower() in ["python"]:
                extensions = [".py"]
                lang_id = "python"
            elif language.lower() in ["java"]:
                extensions = [".java"]
                lang_id = "java"
            else:
                return json.dumps({
                    "error": f"Unsupported language: {language}"
                })
                
            # Find all code files
            all_files = []
            for ext in extensions:
                all_files.extend(dir_path.rglob(f"*{ext}"))
            all_files = [file.relative_to(dir_path) for file in all_files]

            skip_prefixes = ['.', '..', '__', 'node_modules']
            cleaned_files = []
            for file in all_files:
                filename = os.path.basename(file)
                if filename in self.skip_files:
                    continue
                if any([filename.startswith(prefix) for prefix in skip_prefixes]):
                    continue
                if any([str(file).startswith(prefix) for prefix in skip_prefixes]):
                    continue
                cleaned_files.append(file)

            all_files = cleaned_files
            if not all_files:
                return json.dumps({
                    "message": f"No {language} files found in {directory}",
                    "file_count": 0,
                    "diagnostics": []
                })
                
            # Check each file
            all_diagnostics = []
            for file_path in all_files:
                try:
                    content = Path(os.path.join(self.output_dir, str(file_path))).read_text(encoding='utf-8')
                    rel_path = file_path
                    
                    # Convert to absolute path (consistent with other methods)
                    abs_path = Path(self.workspace_dir) / file_path
                    
                    # Open document
                    await server.open_document(str(abs_path), content, lang_id)
                    
                    # Get diagnostics
                    diagnostics = await server.get_diagnostics(str(abs_path))
                    
                    if diagnostics:
                        all_diagnostics.append({
                            "file": str(rel_path),
                            "issues": self._format_diagnostics(diagnostics)
                        })
                except Exception as e:
                    logger.error(f"Error checking file {file_path}: {e}")
                    
            return json.dumps({
                "directory": directory,
                "language": language,
                "file_count": len(all_files),
                "files_with_issues": len(all_diagnostics),
                "diagnostics": all_diagnostics
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error checking directory: {e}")
            return json.dumps({"error": str(e)})
            
    async def _check_code_content(self, content: str, language: str, file_path: Optional[str] = None) -> str:
        """Check code content for errors"""
        try:
            server = await self._get_or_create_server(language)
            if not server:
                return json.dumps({
                    "error": f"Failed to start LSP server for {language}"
                })
                
            # Determine language ID and extension
            if language.lower() in ["typescript", "ts"]:
                lang_id = "typescript"
                ext = ".ts"
            elif language.lower() in ["javascript", "js"]:
                lang_id = "javascript"
                ext = ".js"
            elif language.lower() in ["python", "py"]:
                lang_id = "python"
                ext = ".py"
            elif language.lower() in ["java"]:
                lang_id = "java"
                ext = ".java"
            else:
                return json.dumps({"error": f"Unsupported language: {language}"})
                
            # Use provided file path or create a temp file path
            if file_path:
                check_path = Path(self.workspace_dir) / file_path
            else:
                # Create a temporary file path
                check_path = Path(self.workspace_dir) / f"_temp_check_{os.getpid()}{ext}"
                
            # Open document
            await server.open_document(str(check_path), content, lang_id)
            
            # Get diagnostics
            diagnostics = await server.get_diagnostics(str(check_path))
            
            result = {
                "language": language,
                "has_errors": len(diagnostics) > 0,
                "diagnostic_count": len(diagnostics),
                "diagnostics": self._format_diagnostics(diagnostics)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error checking code content: {e}")
            return json.dumps({"error": str(e)})
            
    async def _update_and_check(self, file_path: str, content: str, language: str) -> str:
        """Update file content and check for errors"""
        try:
            server = await self._get_or_create_server(language)
            if not server:
                return json.dumps({
                    "error": f"Failed to start LSP server for {language}"
                })
                
            # Determine language ID
            if language.lower() in ["typescript", "ts"]:
                lang_id = "typescript"
            elif language.lower() in ["javascript", "js"]:
                lang_id = "javascript"
            elif language.lower() in ["python", "py"]:
                lang_id = "python"
            elif language.lower() in ["java"]:
                lang_id = "java"
            else:
                return json.dumps({"error": f"Unsupported language: {language}"})
                
            full_path = Path(self.workspace_dir) / file_path
            
            # Check if file exists on disk
            file_exists_on_disk = full_path.exists()
            
            # Track version
            if file_path not in self.file_versions:
                # First time opening this file
                self.file_versions[file_path] = 1
                await server.open_document(str(full_path), content, lang_id)
            else:
                # File was opened before
                # If file doesn't exist on disk anymore, it was deleted - close and reopen
                if not file_exists_on_disk:
                    logger.info(f"File {file_path} was deleted, closing old index")
                    await server.close_document(str(full_path))
                    self.file_versions[file_path] = 1
                    await server.open_document(str(full_path), content, lang_id)
                else:
                    # Normal update
                    self.file_versions[file_path] += 1
                    await server.update_document(str(full_path), content, self.file_versions[file_path])
                
            # Get diagnostics
            diagnostics = await server.get_diagnostics(str(full_path))
            
            diagnostics = {
                "file": file_path,
                "language": language,
                "version": self.file_versions[file_path],
                "has_errors": len(diagnostics) > 0,
                "diagnostic_count": len(diagnostics),
                "diagnostics": self._format_diagnostics(diagnostics)
            }

            ignored_errors = [
                'cannot be assigned to', 'is not assignable to',
                'cannot assign to',
                'is unknown', '"none"', 'vue',
                'never used', 'never read', 'implicitly has'
            ]

            if diagnostics.get('has_errors'):
                issues = diagnostics.get('diagnostics', [])
                # Filter critical errors only
                critical_errors = [
                    d for d in issues
                    if d.get('severity') == 'Error' and not any([ignore in d.get('message', '').lower() for ignore in ignored_errors])
                ]

                if critical_errors:
                    error_msg = f"\n⚠️ LSP detected {len(critical_errors)} critical issues:\n"
                    for i, diag in enumerate(critical_errors):
                        line = diag.get('line', 0)
                        msg = diag.get('message', '')
                        error_msg += f"{i}. Line {line}: {msg}\n"
                    return error_msg
            else:
                return ''
            
        except Exception as e:
            logger.error(f"Error updating and checking file: {e}")
            return json.dumps({"error": str(e)})
            
    def _format_diagnostics(self, diagnostics: List[dict]) -> List[dict]:
        """Format diagnostics for better readability"""
        formatted = []
        for diag in diagnostics:
            severity_map = {
                1: "Error",
                2: "Warning",
                3: "Information",
                4: "Hint"
            }
            
            formatted.append({
                "severity": severity_map.get(diag.get("severity", 1), "Error"),
                "message": diag.get("message", ""),
                "line": diag.get("range", {}).get("start", {}).get("line", 0) + 1,
                "column": diag.get("range", {}).get("start", {}).get("character", 0) + 1,
                "source": diag.get("source", ""),
                "code": diag.get("code", "")
            })
            
        return formatted
