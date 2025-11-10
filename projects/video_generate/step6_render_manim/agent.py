import asyncio
import os
import re
import shutil
import subprocess

from moviepy import VideoFileClip
from omegaconf import DictConfig

from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.llm.openai_llm import OpenAI
from ms_agent.utils import get_logger

logger = get_logger()


class RenderManim(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.animation_mode = getattr(self.config, 'animation_code', 'auto')
        self.llm: OpenAI = LLM.from_config(self.config)
        self.fix_history = ''

    async def execute_code(self, inputs, **kwargs):
        messages, context = inputs
        context['foreground_paths'] = []
        segments = context['segments']
        manim_code = context['manim_code']
        logger.info(f'Rendering manim code.')

        async def process_segment(i, segment, code):
            scene_name = f'Scene{i + 1}'
            logger.info(f'Rendering manim code for: {scene_name}')
            scene_dir = os.path.join(self.work_dir, f'scene_{i + 1}')
            if 'manim' in segment:
                os.makedirs(scene_dir, exist_ok=True)
                manim_file = await self.render_manim_scene(code, scene_name, scene_dir, segment, i)
            else:
                manim_file = None
            return manim_file

        tasks = [
            process_segment(i, segment, code)
            for i, (segment, code) in enumerate(zip(segments, manim_code))
        ]

        context['foreground_paths'] = await asyncio.gather(*tasks)
        return messages, context

    async def render_manim_scene(self, code, scene_name, output_dir, segment, i):
        code_file = os.path.join(output_dir, f'{scene_name}.py')
        class_match = re.search(r'class\s+(\w+)\s*\(Scene\)', code)
        actual_scene_name = class_match.group(1) if class_match else scene_name
        output_path = os.path.join(output_dir, f'{scene_name}.mov')
        audio_duration = segment['audio_duration']
        manim_requirement = segment.get('manim')
        class_name = f'Scene{i + 1}'
        content = segment['content']
        final_file_path = None
        if os.path.exists(output_path):
            return output_path
        if manim_requirement is None:
            return None
        logger.info(f'Rendering scene {actual_scene_name}')
        self.fix_history = ''
        for i in range(5):
            with open(code_file, 'w') as f:
                f.write(code)

            env = os.environ.copy()
            env['PYTHONWARNINGS'] = 'ignore'
            env['MANIM_DISABLE_OPENCACHING'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'
            env['LANG'] = 'zh_CN.UTF-8'
            env['LC_ALL'] = 'zh_CN.UTF-8'

            cmd = [
                'manim', 'render', '-ql', '--transparent', '--format=mov',
                '--resolution=1280,720', '--disable_caching',
                os.path.basename(code_file), actual_scene_name
            ]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=output_dir,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    timeout=240,
                    env=env)
                
                output_text = (result.stdout or '') + (result.stderr or '')
            except subprocess.TimeoutExpired as e:
                logger.error(f'Manim rendering timed out after 240 seconds for {actual_scene_name}')
                output_text = (e.stdout.decode('utf-8', errors='ignore') if e.stdout else '') + \
                             (e.stderr.decode('utf-8', errors='ignore') if e.stderr else '')
                logger.error(f'Timeout output: {output_text}')
                # Treat timeout as an error and attempt to fix the code
                code = self.fix_manim_code(
                    output_text,
                    code, manim_requirement, class_name, content, audio_duration
                )
                continue

            if result.returncode != 0:
                logger.warning(f'Manim command exited with code {result.returncode}')
                logger.warning(f'Output: {output_text[:500]}')
                
                real_error_indicators = [
                    'SyntaxError', 'NameError', 'ImportError',
                    'AttributeError', 'TypeError', 'ValueError',
                    'ModuleNotFoundError', 'Traceback', 'Error:',
                    'Failed to render', 'unexpected keyword argument',
                    'got an unexpected', 'invalid syntax'
                ]

                if any([error_indicator in output_text for error_indicator in real_error_indicators]):
                    logger.error(f'Detected real error in Manim output, attempting to fix')
                    code = self.fix_manim_code(output_text, code, manim_requirement, class_name, content, audio_duration)
                    continue

            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file == f'{actual_scene_name}.mov':
                        found_file = os.path.join(root, file)
                        if not RenderManim.verify_and_fix_mov_file(
                                found_file):
                            fixed_path = RenderManim.convert_mov_to_compatible(
                                found_file)
                            if fixed_path:
                                found_file = fixed_path

                        shutil.copy2(found_file, output_path)
                        scaled_path = RenderManim.scale_video_to_fit(
                            output_path, target_size=(1280, 720))
                        if scaled_path and scaled_path != output_path:
                            shutil.rmtree(output_path, ignore_errors=True)
                            shutil.copy2(scaled_path, output_path)
                        final_file_path = output_path
            if not final_file_path:
                code = self.fix_manim_code(output_text, code, manim_requirement, class_name, content, audio_duration)
            else:
                break
        if final_file_path:
            return final_file_path
        else:
            raise FileNotFoundError

    def fix_manim_code(self, error_log, manim_code, manim_requirement, class_name, content, audio_duration):
        fix_request = f"""You are a professional code debugging specialist. You need to help me fix issues in the code. Error messages will be passed directly to you. You need to carefully examine the problems and provide the correct, complete code.
{error_log}

**Original Code**:
```python
{manim_code}
```

{self.fix_history}

**Original code task**: Create manim animation
- Class name: {class_name}
- Content: {content}
- Extra requirement: {manim_requirement}
- Duration: {audio_duration} seconds
- Code language: **Python**

- Please focus on solving the detected issues
- If you find other issues, fix them too
- Keep the good parts, do minimum change, only fix problematic areas
- Ensure no new layout issues are introduced
- If some issues are difficult to solve, prioritize the most impactful ones
- There may be some beginner's error because the code was generated by an AI model
    * Especially pay attention to the quote("" and “”), don't mis-recognize them

Please precisely fix the detected issues while maintaining the richness and creativity of the animation.
"""
        inputs = [Message(role='user', content=fix_request)]
        _response_message = self.llm.generate(inputs)
        response = _response_message.content
        if '```python' in response:
            manim_code = response.split('```python')[1].split('```')[0]
        elif '```' in response:
            manim_code = response.split('```')[1].split('```')[0]
        else:
            manim_code = response
        self.fix_history = (f'You have a fix history which generates the code which is given to you:\n\n{fix_request}\n\n'
                            f'If last error is same with latest error, **You probably find a wrong root cause**, '
                            f'Check carefully and fix it again.**')
        return manim_code

    @staticmethod
    def verify_and_fix_mov_file(mov_path):
        clip = VideoFileClip(mov_path)
        frame = clip.get_frame(0)
        clip.close()
        return frame is not None

    @staticmethod
    def convert_mov_to_compatible(mov_path):
        base_path, ext = os.path.splitext(mov_path)
        fixed_path = f'{base_path}_fixed.mov'
        clip = VideoFileClip(mov_path)
        clip.write_videofile(
            fixed_path,
            codec='libx264',
            audio_codec='aac' if clip.audio else None,
            fps=24,
            verbose=False,
            logger=None,
            ffmpeg_params=['-pix_fmt', 'yuva420p'])

        clip.close()
        if RenderManim.verify_and_fix_mov_file(fixed_path):
            return fixed_path
        else:
            return None

    @staticmethod
    def scale_video_to_fit(video_path, target_size=(1280, 720)):
        if not os.path.exists(video_path):
            return video_path

        clip = VideoFileClip(video_path)
        original_size = clip.size

        target_width, target_height = target_size
        original_width, original_height = original_size

        scale_x = target_width / original_width
        scale_y = target_height / original_height
        scale_factor = min(scale_x, scale_y, 1.0)

        if scale_factor < 0.95:
            scaled_clip = clip.resized(scale_factor)

            base_path, ext = os.path.splitext(video_path)
            scaled_path = f'{base_path}_scaled{ext}'
            scaled_clip.write_videofile(
                scaled_path,
                codec='libx264',
                audio_codec='aac' if scaled_clip.audio else None,
                fps=24,
                verbose=False,
                logger=None)

            clip.close()
            scaled_clip.close()
            return scaled_path
        else:
            return video_path

    def save_history(self, messages, **kwargs):
        messages, context = messages
        self.config.context = context
        return super().save_history(messages, **kwargs)

    def read_history(self, messages, **kwargs):
        _config, _messages = super().read_history(messages, **kwargs)
        if _config is not None:
            context = _config['context']
            return _config, (_messages, context)
        else:
            return _config, _messages