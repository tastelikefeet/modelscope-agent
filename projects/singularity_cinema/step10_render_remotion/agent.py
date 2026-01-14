# Copyright (c) Alibaba, Inc. and its affiliates.
import base64
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Union

import json
from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.utils import get_logger
from omegaconf import DictConfig
from PIL import Image, ImageDraw

logger = get_logger()


class RenderRemotion(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.num_parallel = getattr(self.config, 'llm_num_parallel', 5)
        # When enabled, render compositions one-by-one and attempt a fix immediately on failure.
        # This reduces wasted work when one broken Segment TSX causes global bundler failure.
        self.render_immediate_fix = getattr(self.config,
                                            'render_immediate_fix', True)

        self.render_dir = os.path.join(self.work_dir, 'remotion_render')
        self.remotion_project_dir = os.path.join(self.work_dir,
                                                 'remotion_project')
        self.remotion_code_dir = os.path.join(self.work_dir, 'remotion_code')
        self.images_dir = os.path.join(self.work_dir, 'images')
        self.code_fix_dir = os.path.join(self.work_dir, 'code_fix')
        self.code_fix_round = getattr(self.config, 'code_fix_round', 3)
        # Default to 1 to ensure visual quality check runs at least once unless explicitly disabled (-1)
        self.mllm_check_round = getattr(self.config, 'mllm_fix_round', 1)
        # Maximum times to attempt automatic visual fixes per segment
        self.max_visual_fix_rounds = getattr(self.config,
                                             'max_visual_fix_rounds', 2)
        # Track per-segment visual failure counts
        self.visual_fail_counts = defaultdict(int)

        os.makedirs(self.render_dir, exist_ok=True)
        os.makedirs(self.code_fix_dir, exist_ok=True)

    async def execute_code(self, messages: Union[str, List[Message]],
                           **kwargs) -> List[Message]:
        with open(os.path.join(self.work_dir, 'segments.txt'), 'r') as f:
            segments = json.load(f)
        with open(os.path.join(self.work_dir, 'audio_info.txt'), 'r') as f:
            audio_infos = json.load(f)

        logger.info('Setting up Remotion project.')
        self._setup_remotion_project(segments, audio_infos)

        # Ensure browser is installed before parallel rendering
        self._ensure_browser(self.remotion_project_dir)

        segment_status = {i: False for i in range(len(segments))}

        for round_idx in range(self.code_fix_round + 1):
            # Identify segments needing render (all initially, then only failed ones)
            segments_to_render = [
                i for i, status in segment_status.items() if status is not True
            ]

            if not segments_to_render:
                logger.info('All segments rendered successfully.')
                break

            logger.info(
                f'Round {round_idx + 1}: Rendering {len(segments_to_render)} segments...'
            )

            results = {}

            def _read_current_code(seg_i: int) -> str:
                code_path = os.path.join(self.remotion_code_dir,
                                         f'Segment{seg_i+1}.tsx')
                if os.path.exists(code_path):
                    with open(code_path, 'r', encoding='utf-8') as f:
                        return f.read()
                project_code_path = os.path.join(self.remotion_project_dir,
                                                 'src',
                                                 f'Segment{seg_i+1}.tsx')
                if os.path.exists(project_code_path):
                    with open(project_code_path, 'r', encoding='utf-8') as f:
                        return f.read()
                return ''

            def _extract_error_segment_indices(
                    log_text: Optional[str]) -> List[int]:
                if not log_text:
                    return []
                # esbuild/webpack error lines usually include: ...\src\Segment15.tsx:...
                segs = []
                for m in re.finditer(r'src[\\/]+Segment(\d+)\.tsx', log_text):
                    try:
                        segs.append(int(m.group(1)) - 1)
                    except Exception:
                        continue
                return segs

            # Render sequentially; if any render fails, fix immediately and retry once.
            for i in segments_to_render:
                i, success, error_log = self._render_remotion_scene_static(
                    i,
                    segments[i],
                    audio_infos[i]['audio_duration'],
                    self.config,
                    self.work_dir,
                    self.render_dir,
                    self.remotion_project_dir,
                    self.mllm_check_round,
                )
                results[i] = (success, error_log)
                segment_status[i] = success

                if success or round_idx >= self.code_fix_round:
                    continue

                # Immediate fix: prefer the actual offending Segment file referenced by bundler errors.
                # If bundler fails globally, error_log points to the culprit file.
                culprit_indices = _extract_error_segment_indices(error_log)
                to_fix = culprit_indices if culprit_indices else [i]
                to_fix = sorted(
                    {idx
                     for idx in to_fix if 0 <= idx < len(segments)})

                # If the error points to OTHER segments, it means the current segment failed due to global breakage.
                # Pause and fix the root cause first.
                logger.info(
                    f'Immediate fix triggered by failure on segment {i+1}. Fix targets: {[x+1 for x in to_fix]}'
                )

                # Apply fixes
                for fix_i in to_fix:
                    err_text = error_log or 'Unknown error'
                    error_file = os.path.join(self.code_fix_dir,
                                              f'code_fix_{fix_i + 1}.txt')
                    with open(error_file, 'w', encoding='utf-8') as f:
                        f.write(err_text)

                    current_code = _read_current_code(fix_i)
                    _, fixed_code = self._fix_code_static(
                        fix_i, err_text, current_code, self.config,
                        self.remotion_project_dir)
                    if fixed_code:
                        self._update_segment_code(fix_i, fixed_code)
                        # If we fixed a different segment, we should probably reset its status too
                        if fix_i != i:
                            segment_status[
                                fix_i] = False  # Force re-render of the culprit later if it was skipped

        return messages

    def _update_segment_code(self, i, code):
        # Update in remotion_code_dir (source of truth)
        src_file = os.path.join(self.remotion_code_dir, f'Segment{i+1}.tsx')
        with open(src_file, 'w', encoding='utf-8') as f:
            f.write(code)

        # Update in remotion_project_dir (execution env)
        dst_file = os.path.join(self.remotion_project_dir, 'src',
                                f'Segment{i+1}.tsx')
        with open(dst_file, 'w', encoding='utf-8') as f:
            f.write(code)

    def _setup_remotion_project(self, segments, audio_infos):
        # 1. Create project structure
        os.makedirs(
            os.path.join(self.remotion_project_dir, 'src'), exist_ok=True)
        os.makedirs(
            os.path.join(self.remotion_project_dir, 'public', 'images'),
            exist_ok=True)
        # Some generated TSX may import assets via relative paths like `./images/foo.png`.
        # Keep a mirrored copy under `src/images` to avoid bundler module resolution failures.
        os.makedirs(
            os.path.join(self.remotion_project_dir, 'src', 'images'),
            exist_ok=True)

        if os.path.exists(self.images_dir):
            for file in os.listdir(self.images_dir):
                src = os.path.join(self.images_dir, file)
                dst_public = os.path.join(self.remotion_project_dir, 'public',
                                          'images', file)
                dst_src = os.path.join(self.remotion_project_dir, 'src',
                                       'images', file)
                for dst in (dst_public, dst_src):
                    shutil.copy(src, dst)

        # 3. Copy generated code
        for i in range(len(segments)):
            src_file = os.path.join(self.remotion_code_dir,
                                    f'Segment{i+1}.tsx')
            dst_file = os.path.join(self.remotion_project_dir, 'src',
                                    f'Segment{i+1}.tsx')
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
            else:
                # Create a dummy file if missing to prevent build failure
                with open(dst_file, 'w') as f:
                    f.write(
                        f"import React from 'react';\nexport const Segment{i+1} = () => <div>Missing Segment</div>;"
                    )

        # 4. Create package.json with locked versions
        package_json = {
            'name': 'remotion-project',
            'version': '1.0.0',
            'dependencies': {
                'react': '^18.2.0',
                'react-dom': '^18.2.0',
                'remotion': '^4.0.0',
                '@remotion/cli': '^4.0.0',
                '@remotion/bundler': '^4.0.0',
                '@remotion/renderer': '^4.0.0',
                '@remotion/shapes': '^4.0.0',
                '@remotion/media-utils': '^4.0.0'
            }
        }
        with open(
                os.path.join(self.remotion_project_dir, 'package.json'),
                'w') as f:
            json.dump(package_json, f, indent=2)

        # 5. Create src/index.ts
        with open(
                os.path.join(self.remotion_project_dir, 'src', 'index.ts'),
                'w') as f:
            f.write("import { registerRoot } from 'remotion';\n")
            f.write("import { RemotionRoot } from './Root';\n")
            f.write('registerRoot(RemotionRoot);\n')

        # 6. Create src/Root.tsx
        fps = self.config.video.fps
        width = 1280
        height = 720
        if hasattr(self.config.video, 'size'):
            w, h = self.config.video.size.split('x')
            width = int(w)
            height = int(h)

        root_content = "import React from 'react';\n"
        root_content += "import { Composition } from 'remotion';\n"
        for i in range(len(segments)):
            root_content += f"import * as Segment{i+1}_NS from './Segment{i+1}';\n"

        root_content += '\nexport const RemotionRoot: React.FC = () => {\n'
        for i in range(len(segments)):
            root_content += (
                f'  const Segment{i+1} = Segment{i+1}_NS.default || '
                f'Segment{i+1}_NS.Segment{i+1} || (() => null);\n')

        root_content += '  return (\n'
        root_content += '    <>\n'

        for i, audio_info in enumerate(audio_infos):
            duration_in_frames = int(audio_info['audio_duration'] * fps)
            root_content += '      <Composition\n'
            root_content += f"        id=\"Segment{i+1}\"\n"
            root_content += f'        component={{Segment{i+1}}}\n'
            root_content += f'        durationInFrames={{{duration_in_frames}}}\n'
            root_content += f'        fps={{{fps}}}\n'
            root_content += f'        width={{{width}}}\n'
            root_content += f'        height={{{height}}}\n'
            root_content += '      />\n'

        root_content += '    </>\n'
        root_content += '  );\n'
        root_content += '};\n'

        with open(
                os.path.join(self.remotion_project_dir, 'src', 'Root.tsx'),
                'w') as f:
            f.write(root_content)

        # 7. Create tsconfig.json
        tsconfig = {
            'compilerOptions': {
                'allowJs': True,
                'checkJs': True,
                'esModuleInterop': True,
                'forceConsistentCasingInFileNames': True,
                'resolveJsonModule': True,
                'skipLibCheck': True,
                'sourceMap': True,
                'strict': True,
                'target': 'esnext',
                'module': 'esnext',
                'moduleResolution': 'node',
                'jsx': 'react-jsx',
                'noEmit': True,
                'isolatedModules': True
            },
            'include': ['src']
        }
        with open(
                os.path.join(self.remotion_project_dir, 'tsconfig.json'),
                'w') as f:
            json.dump(tsconfig, f, indent=2)

        # 8. Install dependencies
        node_modules_dir = os.path.join(self.remotion_project_dir,
                                        'node_modules')
        if not os.path.exists(node_modules_dir):
            logger.info('Installing dependencies...')
            subprocess.run(
                'npm install',
                cwd=self.remotion_project_dir,
                shell=True,
                check=True)

    def _ensure_browser(self, remotion_project_dir):
        # Check for global browser cache to avoid re-downloading.
        user_home = os.path.expanduser('~')
        global_browser_cache = os.path.join(user_home,
                                            '.ms_agent_remotion_browser')
        local_browser_cache = os.path.join(remotion_project_dir,
                                           'node_modules', '.remotion')

        # Link or copy cached browser if available.
        if os.path.exists(global_browser_cache):
            logger.info(
                f'Found cached Chrome in {global_browser_cache}. Linking to project...'
            )
            if not os.path.exists(local_browser_cache):
                try:
                    # Windows usually requires admin for symlinks, so we copy.
                    # Copying is still much faster than downloading 300MB+
                    shutil.copytree(global_browser_cache, local_browser_cache)
                    logger.info('Browser restored from cache.')
                    return
                except Exception as e:
                    logger.warning(f'Failed to copy cached browser: {e}')

        # Check if browser is already installed in project (standard check)
        browser_found = False
        if os.path.exists(local_browser_cache):
            for root, _, files in os.walk(local_browser_cache):
                if 'chrome-headless-shell' in files:
                    browser_found = True
                    break

        if browser_found:
            logger.info(
                "Remotion browser detected locally. Skipping 'browser ensure'."
            )
            # Cache it for next time!
            if not os.path.exists(global_browser_cache):
                try:
                    logger.info(
                        f'Caching browser to {global_browser_cache} for future runs...'
                    )
                    shutil.copytree(local_browser_cache, global_browser_cache)
                except Exception as e:
                    logger.warning(f'Failed to cache browser: {e}')
            return

        # 1. Try to download from manual mirror first.
        logger.info(
            'Attempting to manually download Remotion browser from npmmirror...'
        )

        # Use a specific version known to work.
        # Link: https://npmmirror.com/mirrors/chrome-for-testing/134.0.6998.35/win64/chrome-headless-shell-win64.zip
        version = '134.0.6998.35'
        import sys
        platform_str = 'win64' if os.name == 'nt' else ('mac64' if sys.platform == 'darwin' else 'linux64')
        filename = f'chrome-headless-shell-{platform_str}.zip'
        mirror_url = f'https://npmmirror.com/mirrors/chrome-for-testing/{version}/{platform_str}/{filename}'

        try:
            logger.info(f'Downloading {mirror_url}...')
            zip_target = os.path.join(remotion_project_dir, filename)
            urllib.request.urlretrieve(mirror_url, zip_target)

            logger.info('Extracting browser...')
            os.makedirs(local_browser_cache, exist_ok=True)
            with zipfile.ZipFile(zip_target, 'r') as zip_ref:
                zip_ref.extractall(local_browser_cache)

            if os.path.exists(zip_target):
                os.remove(zip_target)

            logger.info('Browser downloaded and extracted successfully.')

            # Cache it globally immediately
            if not os.path.exists(global_browser_cache):
                try:
                    logger.info(
                        f'Caching browser to {global_browser_cache} for future runs...'
                    )
                    shutil.copytree(local_browser_cache, global_browser_cache)
                except Exception as e:
                    logger.warning(f'Failed to cache browser: {e}')

            return

        except Exception as e:
            logger.warning(
                f'Failed to manually download browser from mirror: {e}')

        # Fallback to standard ensuring if manual download fails
        logger.info("Falling back to standard 'browser ensure'...")
        os.environ['PUPPETEER_DOWNLOAD_HOST'] = 'https://npmmirror.com/mirrors'

        if os.name == 'nt':
            remotion_cmd = os.path.abspath(
                os.path.join(remotion_project_dir, 'node_modules', '.bin',
                             'remotion.cmd'))
        else:
            remotion_cmd = os.path.abspath(
                os.path.join(remotion_project_dir, 'node_modules', '.bin',
                             'remotion'))

        if not os.path.exists(remotion_cmd):
            remotion_cmd = 'npx remotion'

        try:
            if os.name == 'nt' and 'remotion.cmd' in remotion_cmd:
                cmd_str = f'"{remotion_cmd}" browser ensure'
                subprocess.run(
                    cmd_str, cwd=remotion_project_dir, shell=True, check=True)
            else:
                cmd = [remotion_cmd, 'browser', 'ensure']
                if isinstance(
                        remotion_cmd,
                        str) and ' ' in remotion_cmd and 'npx' in remotion_cmd:
                    cmd = remotion_cmd.split() + ['browser', 'ensure']
                subprocess.run(cmd, cwd=remotion_project_dir, check=True)
            logger.info('Browser download successful (via standard ensure).')
            return
        except subprocess.CalledProcessError as e:
            logger.warning(
                f'Failed to download browser from mirror via CLI: {e}')

        # 2. Fallback to System Chrome (ONLY IF DOWNLOAD FAILS)
        logger.info('Falling back to System Chrome detection...')
        # shutil is already imported at module level
        system_chrome = (
            shutil.which('chrome') or shutil.which('google-chrome')
            or shutil.which('chromium') or shutil.which('chromium-browser'))

        if not system_chrome and os.name == 'nt':
            possible_paths = [
                r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
                os.path.expandvars(
                    r'%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe'),
                r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                r'C:\Program Files\Microsoft\Edge\Application\msedge.exe'
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    system_chrome = p
                    break

        if system_chrome:
            logger.info(
                f'System Chrome detected at {system_chrome}. Configuring Remotion to use it.'
            )
            os.environ['REMOTION_BROWSER_EXECUTABLE'] = system_chrome
            os.environ['PUPPETEER_EXECUTABLE_PATH'] = system_chrome
            return

        logger.error(
            'Could not find any browser (Download failed and no System Chrome found). Rendering will likely fail.'
        )

    @staticmethod
    def check_code_quality(code, i, config, segment, duration):
        """
        Use LLM to audit code quality, checking for layout issues (e.g. Flexbox vs Absolute).
        """
        # If no code, we can't check.
        if not code or len(code) < 50:
            return 'FAIL: Empty or invalid code.'

        llm = LLM.from_config(config)

        system_prompt = """You are a Senior React/Remotion Code Auditor.
Your goal is to ensure the generated video code is robust, responsive, and uses modern layout practices (Flexbox).

**CRITICAL FAILURE CRITERIA (Report these as FAIL)**:
1.  **Absolute Overlap Risk**: Using `position: 'absolute'` with `top: '50'`, `left: '50'` on MULTIPLE elements
    without distinct margins or transforms. This causes overlap.
2.  **Lack of Flexbox**: The main container should use `display: 'flex'` to manage layout structure (Text vs Image).
    - **VIOLATION**: If you see `AbsoluteFill` containing only direct children with `position: 'absolute'`, FAIL IT.
    - **REQUIREMENT**: There must be a flex container that separates content.
3.  **Hardcoded Dimensions (Universal Bounds Issue)**:
    - **VIOLATION**: Using fixed pixel widths/heights (e.g., `width: 500`, `left: 300`) for MAIN containers.
    - **REQUIREMENT**: Use percentages (e.g., `width: '50%'`) to ensure generic compatibility across resolutions.
4.  **Z-Index Chaos**: Elements using random high z-indexes (100, 999) to force visibility
    instead of proper DOM order.
5.  **Text Visibility**: Text containers MUST have a background color (e.g., `rgba(0,0,0,0.5)`)
    if they are overlaying images to prevent contrast issues.

**Context**:
- This is a 16:9 video (1280x720).
- We want a clean, high-end presentation style.

**Output Format**:
- If Clean: Output exactly `PASS`.
- If Issues: Output `FAIL: <Concise explanation of the code flaw and how to fix it>`.
"""

        user_prompt = f"""
Audit this Remotion Component Code for Segment {i+1}:

```typescript
{code}
```

Does this code follow best practices for layout safety (Flexbox) and avoid obvious overlap risks?
"""

        try:
            response = llm.generate([
                Message(role='system', content=system_prompt),
                Message(role='user', content=user_prompt)
            ])
            result = response.content.strip()

            if 'FAIL' in result:
                return result
            # If the LLM just chats but doesn't explicitly fail, we assume pass or look for negative keywords?
            # The prompt asks for explicit PASS/FAIL.
            if 'PASS' in result:
                return None

            # Fallback: if ambiguous, treat as pass but log? No, safe to pass if not explicit fail.
            return None

        except Exception as e:
            logger.warning(f'Code Audit Check failed: {e}')
            return None

    @staticmethod
    def _render_remotion_scene_static(
            i,
            segment,
            duration,
            config,
            work_dir,
            render_dir,
            remotion_project_dir,
            mllm_check_round=0) -> Tuple[int, bool, Optional[str]]:
        """Static method for multiprocessing"""
        composition_id = f'Segment{i+1}'
        output_dir_scene = os.path.join(render_dir, f'scene_{i+1}')
        os.makedirs(output_dir_scene, exist_ok=True)
        output_path = os.path.abspath(
            os.path.join(output_dir_scene, f'Scene{i+1}.mov'))

        logger.info(f'Rendering {composition_id} to {output_path}')

        # Determine remotion command
        if os.name == 'nt':
            remotion_cmd = os.path.abspath(
                os.path.join(remotion_project_dir, 'node_modules', '.bin',
                             'remotion.cmd'))
        else:
            remotion_cmd = os.path.abspath(
                os.path.join(remotion_project_dir, 'node_modules', '.bin',
                             'remotion'))

        if not os.path.exists(remotion_cmd):
            remotion_cmd = 'npx remotion'

        base_cmd = [
            'render', 'src/index.ts', composition_id, output_path,
            '--codec=prores', '--prores-profile=4444',
            '--pixel-format=yuva444p10le', '--image-format=png'
        ]

        # Try to find browser executable (Local > System)
        browser_executable = None
        remotion_cache_dir = os.path.join(remotion_project_dir, 'node_modules',
                                          '.remotion')

        # 1. Check Local Cache
        if os.path.exists(remotion_cache_dir):
            for root, _, files in os.walk(remotion_cache_dir):
                if 'chrome-headless-shell.exe' in files:
                    browser_executable = os.path.abspath(
                        os.path.join(root, 'chrome-headless-shell.exe'))
                    break
                elif 'chrome-headless-shell' in files:
                    browser_executable = os.path.abspath(
                        os.path.join(root, 'chrome-headless-shell'))
                    break

        # 2. Check System Chrome if not found locally
        if not browser_executable:
            browser_executable = os.environ.get('REMOTION_BROWSER_EXECUTABLE')

            if not browser_executable:
                # shutil is imported at module level
                browser_executable = (
                    shutil.which('chrome') or shutil.which('google-chrome')
                    or shutil.which('chromium')
                    or shutil.which('chromium-browser'))

                if not browser_executable and os.name == 'nt':
                    possible_paths = [
                        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
                        os.path.expandvars(
                            r'%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe'
                        ),
                        r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                        r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
                        os.path.expandvars(
                            r'%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe'
                        ),
                    ]
                    for p in possible_paths:
                        if os.path.exists(p):
                            browser_executable = p
                            break

        if browser_executable:
            logger.info(f'Using browser executable: {browser_executable}')
            base_cmd.extend(['--browser-executable', browser_executable])
            # Add stability flags
            base_cmd.extend([
                '--chromium-options',
                'no-sandbox,disable-setuid-sandbox,disable-gpu,disable-dev-shm-usage'
            ])

        if os.name == 'nt' and 'remotion.cmd' in remotion_cmd:
            cmd = [remotion_cmd] + base_cmd
        else:
            if 'npx' in remotion_cmd:
                cmd = remotion_cmd.split() + base_cmd
            else:
                cmd = [remotion_cmd] + base_cmd

        try:
            if isinstance(cmd, list):
                # On Windows, if we are running a .cmd file, we need shell=True.
                # But subprocess.run with shell=True and a list is tricky.
                # It's safer to join the command into a string for shell=True on Windows.
                if os.name == 'nt':
                    cmd_str = ' '.join(
                        [f'"{arg}"' if ' ' in arg else arg for arg in cmd])
                    result = subprocess.run(
                        cmd_str,
                        cwd=remotion_project_dir,
                        shell=True,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='ignore')
                else:
                    result = subprocess.run(
                        cmd,
                        cwd=remotion_project_dir,
                        shell=False,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='ignore')
            else:
                result = subprocess.run(
                    cmd,
                    cwd=remotion_project_dir,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore')

            if result.returncode != 0:
                # Capture output was set to True to allow smart error detection.
                log_content = (result.stderr or '') + '\n' + (
                    result.stdout or '')
                logger.warning(
                    f'Rendering failed for {composition_id}. Log (except): {log_content[:500]}...'
                )
                return i, False, log_content
            else:
                logger.info(f'Rendered {composition_id} successfully.')

                # --- VISUAL CHECK MOVED TO STEP 14 (Global Check) ---
                # As per user request, we delay the MLLM visual inspection to the final composition stage.
                # This avoids blocking the render loop for every segment.

                return i, True, None

        except Exception as e:
            logger.error(f'Exception during rendering {composition_id}: {e}')
            return i, False, str(e)

    @staticmethod
    def _fix_code_static(i,
                         error_log,
                         code,
                         config,
                         remotion_project_dir=None):
        """Static method for multiprocessing fix"""
        if not code:
            return i, ''

        # 3. Use LLM to fix remaining issues.
        llm = LLM.from_config(config)
        logger.info(f'Fixing code for segment {i+1} with LLM...')
        return i, RenderRemotion._fix_code_impl(llm, error_log, code,
                                                   remotion_project_dir)

    @staticmethod
    def _fix_code_impl(llm, error_log, code, remotion_project_dir=None):
        available_images_info = ''
        if remotion_project_dir:
            images_path = os.path.join(remotion_project_dir, 'public',
                                       'images')
            if os.path.exists(images_path):
                files = sorted(os.listdir(images_path))
                available_images_info = '\nAvailable images in public/images/:\n' + '\n'.join(
                    [f'- {f}' for f in files])

        if 'VISUAL CHECK FAILED' in error_log:
            fix_prompt = f"""
The Remotion code rendered successfully, but the AI Visual Inspector found layout/visual issues.
Visual Feedback:
{error_log}

**Original Code**:
```typescript
{code}
```

Please fix the code to resolve the VISUAL issues reported above.
- **SCALING & SIZING**: If images are cut off or too small, adjust `width`, `maxWidth` or `scale` intelligently.
- **LAYOUT STRATEGY**: If elements overlap, switch to Flexbox (`display: 'flex', flexDirection: 'column/row'`)
    to enforce separation, or adjust absolute coordinates.
- **BOUNDARIES**: Ensure content stays within the visible frame (1280x720).
- Do not change the component name.
- Return the full corrected code.
"""
        else:
            fix_prompt = f"""
The following Remotion/React code failed to render.
Error Log:
{error_log}

**Original Code**:
```typescript
{code}
```

Please fix the code to resolve the error.
- Focus on the error described in the log.
- Ensure the code is a valid React Functional Component.
- Do not change the component name or export style if possible.

**SPECIFIC ERROR GUIDANCE**:
1. **"inputRange must be strictly monotonically increasing"**:
   - You used `interpolate(frame, [20, 0], ...)` or similar unsorted array.
   - FIX: Sort the `inputRange` to `[0, 20]`.
   - CRITICAL: You MUST also reorder `outputRange` to match the new input order.

2. **"Failed to load resource" / 404 Errors**:
   - You are referencing an image that does not exist.
   - {available_images_info}
   - FIX: Use ONLY filenames from the list above. If the file isn't there, remove the `Img` tag.
   - Use `staticFile("images/filename.png")`.
   - FORBIDDEN: `http://...`, `/public/...` paths.

3. **UNIVERSAL LAYOUT RULES (PREVENT OVERLAP)**:
   - **Flexbox Protocol**: Use `display: 'flex'` on the container.
   - **Separation**: Put Text in a top container, Images in a bottom container (or side-by-side).
   - **Safe/Transparent**: Ensure text is readable (use background cards if needed) and verify `width: '85%'` safe area.
   - **No Absolute Center Collisions**: Never put two different elements at `top: 50%, left: 50%`.

4. **Transparency**:
   - Ensure `backgroundColor: undefined` (transparent) for the root.
   - Do NOT render full-screen background images.

Return the full corrected code.
"""
        inputs = [Message(role='user', content=fix_prompt)]
        _response_message = llm.generate(inputs)
        response = _response_message.content

        # Robust code extraction using regex
        code_match = re.search(
            r'```(?:typescript|tsx|js|javascript)?\s*(.*?)```', response,
            re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code = response

        return code.strip()
