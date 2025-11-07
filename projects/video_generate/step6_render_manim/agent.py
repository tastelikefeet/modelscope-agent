import os
import re
import shutil
import subprocess
import tempfile

from moviepy import VideoFileClip
from omegaconf import DictConfig

from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM
from ms_agent.llm.openai_llm import OpenAI


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

    async def execute_code(self, inputs, **kwargs):
        messages, context = inputs
        context['foreground_paths'] = []
        segments = context['segments']
        manim_code = context['manim_code']
        for i, (segment, code) in enumerate(segments, manim_code):
            if segment['type'] == 'text' and self.animation_mode == 'human':
                context['foreground_paths'].append(None)
                continue
            scene_name = f'Scene{i + 1}'
            scene_dir = os.path.join(self.work_dir, f'scene_{i + 1}')
            manim_file = self.render_manim_scene(code, scene_name, scene_dir)
            context['foreground_paths'].append(manim_file)

    @staticmethod
    def render_manim_scene(code, scene_name, output_dir):
        code_file = os.path.join(output_dir, f'{scene_name}.py')
        with open(code_file, 'w') as f:
            f.write(code)
        class_match = re.search(r'class\s+(\w+)\s*\(Scene\)', code)
        actual_scene_name = class_match.group(1) if class_match else scene_name
        output_path = os.path.join(output_dir, f'{scene_name}.mov')

        with tempfile.TemporaryDirectory() as temp_dir:
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

            result = subprocess.run(
                cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=300,
                env=env)

            output_text = (result.stdout or '') + (result.stderr or '')
            if result.returncode == 1:

                real_error_indicators = [
                    'SyntaxError', 'NameError', 'ImportError',
                    'AttributeError', 'TypeError', 'ValueError',
                    'ModuleNotFoundError', 'Traceback', 'Error:',
                    'Failed to render'
                ]

                for error_indicator in real_error_indicators:
                    if error_indicator in output_text:
                        break

            temp_media_dir = os.path.join(temp_dir, 'media', 'videos')
            assert os.path.exists(temp_media_dir)
            for root, dirs, files in os.walk(temp_media_dir):
                for file in files:
                    if file == f'{actual_scene_name}.mov':
                        found_file = os.path.join(root, file)
                        shutil.copy2(found_file, output_path)
                        if not RenderManim.verify_and_fix_mov_file(
                                output_path):
                            fixed_path = RenderManim.convert_mov_to_compatible(
                                output_path)
                            if fixed_path:
                                output_path = fixed_path

                        scaled_path = RenderManim.scale_video_to_fit(
                            output_path, target_size=(1280, 720))
                        if scaled_path and scaled_path != output_path:
                            return scaled_path

                        return output_path
            raise FileNotFoundError

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
            scaled_clip = clip.resize(scale_factor)

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