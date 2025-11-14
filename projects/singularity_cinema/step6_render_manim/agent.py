# Copyright (c) Alibaba, Inc. and its affiliates.
import base64
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union

import json
from moviepy import VideoFileClip
from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


class RenderManim(CodeAgent):

    test_system = """你是一个帮助检查动画布局问题的专家。你会被给与一张图片，该图片可能是某个完整manim动画的中间帧或最后一帧。你需要帮助给出该图片动画中所有不合理的布局之处。

你在一个短视频制作的工作流程中，该流程大致为：
1. LLM生成文字台本和分镜台本，其中分镜台本包含了约5~10秒的独白和动画要求。
2. LLM根据动画要求生成manim代码，并渲染成为mov文件，给你的图片就是该mov文件中的一帧。
    * 短视频大小为1920*1080，但可用渲染范围为1500*750，其余部分留作它用

[CRITICAL]你需要关注的不足之处：
1. 是否有组件重叠或截断，【重要】你需要特别关注图像边缘
    * [严重！]组件、文字重叠
    * [严重！]组件、文字被图片边缘切断，展示不完整（即使是轻微显示不完整！）
2. 是否有组件位置不合理，例如：
    * 同一功能的两个组件上下不对齐，左右不等高
    * [严重！]子母组件不协调，例如子组件或文字没有完整位于内部，而是和外部母组件重叠或超出边框
    * [严重！]饼图圆心不一致导致没有展示为一个完整的圆，或直方图、折线图等位置错误
    * [严重！]组件不对称，位于屏幕右侧或左侧，另一侧完全空白，或组件太小太大，不协调
    * 连接组件的线起点终点错误，或箭头方向错误，以及线和组件重叠等问题

你需要详细描述哪个组件有什么样的问题，你不需要给出修复意见，但你需要尽可能真实描述问题现象和位置。

针对问题的严重程度，分级如下：
1. 组件被截断、组件重叠、不对齐、位置不合理问题，此类问题必须反馈
2. 不美观等问题不要反馈，防止动画代码生成的死锁问题
3. 如果是中间帧，这张图片极大可能没有展现完整，因此你**不需要关注不完整**问题，仅需要关注重叠问题，忽略孤立或未完成的组件；如果是最后一帧，你需要关注上面提到的所有问题

你反馈的问题必须要包装在<result>问题列表</result>中，如果没有发现明显问题，应当返回<result></result>，即内部为空内容。
下面开始：
"""

    window_size = (1800, 900)

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.num_parallel = getattr(self.config, 'llm_num_parallel', 10)
        self.manim_render_timeout = getattr(self.config,
                                            'manim_render_timeout', 300)
        self.render_dir = os.path.join(self.work_dir, 'manim_render')
        os.makedirs(self.render_dir, exist_ok=True)

    async def execute_code(self, messages: Union[str, List[Message]],
                           **kwargs) -> List[Message]:
        with open(os.path.join(self.work_dir, 'segments.txt'), 'r') as f:
            segments = json.load(f)
        manim_code_dir = os.path.join(self.work_dir, 'manim_code')
        manim_code = []
        for i in range(len(segments)):
            with open(os.path.join(manim_code_dir, f'segment_{i+1}.py'),
                      'r') as f:
                manim_code.append(f.read())
        with open(os.path.join(self.work_dir, 'audio_info.txt'), 'r') as f:
            audio_infos = json.load(f)
        assert len(manim_code) == len(segments)
        logger.info('Rendering manim code.')

        tasks = [
            (i, segment, code, audio_info['audio_duration'])
            for i, (segment, code, audio_info
                    ) in enumerate(zip(segments, manim_code, audio_infos))
        ]

        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = {
                executor.submit(self._render_manim_scene_static, i, segment,
                                code, duration, self.config, self.work_dir,
                                self.render_dir, self.window_size,
                                self.manim_render_timeout): i
                for i, segment, code, duration in tasks
            }
            for future in as_completed(futures):
                future.result()  # Wait for completion and raise any exceptions

        return messages

    @staticmethod
    def _render_manim_scene_static(i, segment, code, audio_duration, config,
                                   work_dir, render_dir, window_size,
                                   manim_render_timeout):
        """Static method for multiprocessing"""
        llm = LLM.from_config(config)
        return RenderManim._render_manim_impl(llm, i, segment, code,
                                              audio_duration, work_dir,
                                              render_dir, window_size,
                                              manim_render_timeout, config)

    @staticmethod
    def _render_manim_impl(llm, i, segment, code, audio_duration, work_dir,
                           render_dir, window_size, manim_render_timeout, config):
        scene_name = f'Scene{i+1}'  # sometimes actual_scene_name cannot find matched class, so do not change this name
        logger.info(f'Rendering manim code for: scene_{i + 1}')
        output_dir = os.path.join(render_dir, f'scene_{i + 1}')
        os.makedirs(output_dir, exist_ok=True)
        if 'manim' not in segment:
            return None
        code_file = os.path.join(output_dir, f'{scene_name}.py')
        class_match = re.search(r'class\s+(\w+)\s*\(Scene\)', code)
        actual_scene_name = class_match.group(1) if class_match else scene_name
        output_path = os.path.join(output_dir, f'{scene_name}.mov')
        manim_requirement = segment.get('manim')
        class_name = f'Scene{i + 1}'
        content = segment['content']
        final_file_path = None
        if os.path.exists(output_path):
            return output_path
        logger.info(f'Rendering scene {actual_scene_name}')
        fix_history = ''
        mllm_max_check_round = 3
        cur_check_round = 0
        for retry_idx in range(10):
            with open(code_file, 'w') as f:
                f.write(code)

            env = os.environ.copy()
            env['PYTHONWARNINGS'] = 'ignore'
            env['MANIM_DISABLE_OPENCACHING'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'
            env['LANG'] = 'zh_CN.UTF-8'
            env['LC_ALL'] = 'zh_CN.UTF-8'
            window_size_str = ','.join([str(x) for x in window_size])
            cmd = [
                'manim', 'render', '-ql', '--transparent', '--format=mov',
                f'--resolution={window_size_str}', '--disable_caching',
                os.path.basename(code_file), actual_scene_name
            ]

            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=output_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    env=env)

                # Wait for process to complete with timeout
                stdout, stderr = process.communicate(
                    timeout=manim_render_timeout)

                # Create result object compatible with original logic
                class Result:

                    def __init__(self, returncode, stdout, stderr):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                result = Result(process.returncode, stdout, stderr)
                output_text = (result.stdout or '') + (result.stderr or '')
            except subprocess.TimeoutExpired as e:
                output_text = (e.stdout.decode('utf-8', errors='ignore')
                               if e.stdout else '') + (
                                   e.stderr.decode('utf-8', errors='ignore')
                                   if e.stderr else '')  # noqa
                logger.error(
                    f'Manim rendering timed out after {manim_render_timeout} '
                    f'seconds for {actual_scene_name}, output: {output_text}')
                logger.info('Trying to fix manim code.')
                code, fix_history = RenderManim._fix_manim_code_impl(
                    llm, output_text, fix_history, code, manim_requirement,
                    class_name, content, audio_duration)
                continue

            if result.returncode != 0:
                logger.warning(
                    f'Manim command exited with code {result.returncode}')
                logger.warning(f'Output: {output_text}')

                real_error_indicators = [
                    'SyntaxError', 'NameError', 'ImportError',
                    'AttributeError', 'TypeError', 'ValueError',
                    'ModuleNotFoundError', 'Traceback', 'Error:',
                    'Failed to render', 'unexpected keyword argument',
                    'got an unexpected', 'invalid syntax'
                ]

                if any([
                        error_indicator in output_text
                        for error_indicator in real_error_indicators
                ]):
                    logger.info('Trying to fix manim code.')
                    code, fix_history = RenderManim._fix_manim_code_impl(
                        llm, output_text, fix_history, code, manim_requirement,
                        class_name, content, audio_duration)
                    continue

            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file == f'{actual_scene_name}.mov':
                        found_file = os.path.join(root, file)
                        if not RenderManim.verify_and_fix_mov_file(found_file):
                            fixed_path = RenderManim.convert_mov_to_compatible(
                                found_file)
                            if fixed_path:
                                found_file = fixed_path

                        shutil.copy2(found_file, output_path)
                        scaled_path = RenderManim.scale_video_to_fit(
                            output_path, target_size=window_size)
                        if scaled_path and scaled_path != output_path:
                            shutil.rmtree(output_path, ignore_errors=True)
                            shutil.copy2(scaled_path, output_path)
                        final_file_path = output_path
            if not final_file_path:
                logger.error(
                    f'Manim file: {class_name} not found, trying to fix manim code.'
                )
                code, fix_history = RenderManim._fix_manim_code_impl(
                    llm, output_text, fix_history, code, manim_requirement,
                    class_name, content, audio_duration)
            else:
                if cur_check_round >= mllm_max_check_round:
                    break
                output_text = RenderManim.check_manim_quality(final_file_path, work_dir, i, config)
                cur_check_round += 1
                if output_text:
                    try:
                        os.remove(final_file_path)
                        final_file_path = None
                    except OSError:
                        pass
                    logger.info(f'Trying to fix manim code of segment {i+1}, because model checking not passed: \n{output_text}')
                    code, fix_history = RenderManim._fix_manim_code_impl(
                        llm, output_text, fix_history, code, manim_requirement,
                        class_name, content, audio_duration)
                    continue
                else:
                    break
        if final_file_path:
            RenderManim._extract_preview_frames_static(final_file_path, i, work_dir)
            manim_code_dir = os.path.join(work_dir, 'manim_code')
            manim_file = os.path.join(manim_code_dir, f'segment_{i + 1}.py')
            with open(manim_file, 'w') as f:
                f.write(code)
        else:
            raise FileNotFoundError(final_file_path)

    @staticmethod
    def check_manim_quality(final_file_path, work_dir, i, config):
        _mm_config = DictConfig({
            'llm': {
                'service': 'openai',
                'model': config.manim_auto_test.manim_test_model,
                'openai_api_key': config.manim_auto_test.manim_test_api_key,
                'openai_base_url': config.manim_auto_test.manim_test_base_url,
            },
            'generation_config': {
                'temperature': 0.3
            }
        })
        test_images = RenderManim._extract_preview_frames_static(final_file_path, i, work_dir)
        llm = LLM.from_config(_mm_config)

        frame_names = ['中间帧', '最后一帧']

        all_issues = []
        for idx, (image_path, frame_name) in enumerate(zip(test_images, frame_names)):
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')

            content = [
                {
                    "type": "text",
                    "text": f"当前分析的是{frame_name}，请仔细检查该帧中的动画布局问题。"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]

            messages = [
                Message(role='system', content=RenderManim.test_system),
                Message(role='user', content=content),
            ]
            response = llm.generate(messages)
            response_text = response.content

            pattern = r'<result>(.*?)</result>'
            issues = []
            for issue in re.findall(pattern, response_text, re.DOTALL):
                issues.append(issue)
            issues = '\n'.join(issues).strip()
            if issues:
                issues = f'Current is the {frame_name}, problem checked by a MLLM: {issues}'
            all_issues.append(issues)

        all_issues = '\n'.join(all_issues).strip()
        if all_issues:
            all_issues = ('The middle and last frame of the rendered animation was sent to a multi-modal LLM to check '
                          f'the layout problems, and here are the possible issues:\n{all_issues}, '
                          f'Some of the MLLM\'s feedback is incorrect. But if `cutt off` or `overlap` issues are mentioned, you need to carefully check your code '
                          f'about it, then determine which issues actually need to be fixed.')
        return all_issues

    @staticmethod
    def _extract_preview_frames_static(video_path, segment_id, work_dir):
        from moviepy import VideoFileClip
        
        test_dir = os.path.join(work_dir, 'manim_test')
        os.makedirs(test_dir, exist_ok=True)
        video = VideoFileClip(video_path)
        duration = video.duration

        timestamps = {
            1: duration / 2,
            2: max(0, duration - 0.2)
        }

        preview_paths = []
        for frame_idx, timestamp in timestamps.items():
            output_path = os.path.join(
                test_dir,
                f'segment_{segment_id + 1}_{frame_idx}.png'
            )
            video.save_frame(output_path, t=timestamp)
            preview_paths.append(output_path)
        video.close()
        return preview_paths

    @staticmethod
    def _fix_manim_code_impl(llm, error_log, fix_history, manim_code,
                             manim_requirement, class_name, content,
                             audio_duration):
        fix_request = f"""You are a professional code debugging specialist. You need to help me fix issues in the code. Error messages will be passed directly to you. You need to carefully examine the problems and provide the correct, complete code.
{error_log}

**Original Code**:
```python
{manim_code}
```

{fix_history}

**Original code task**: Create manim animation
- Class name: {class_name}
- Content: {content}
- Extra requirement: {manim_requirement}
- Duration: {audio_duration} seconds
- Code language: **Python**

- Please focus on solving the detected issues
- If you find other issues, fix them too
- Keep the good parts, do minimum changes, only fix problematic areas
- Ensure no new layout issues are introduced
- If some issues are difficult to solve, prioritize the most impactful ones
- There may be some beginner's error because the code was generated by an AI model

Please precisely fix the detected issues while maintaining the richness and creativity of the animation.
""" # noqa
        inputs = [Message(role='user', content=fix_request)]
        _response_message = llm.generate(inputs)
        response = _response_message.content
        if '```python' in response:
            manim_code = response.split('```python')[1].split('```')[0]
        elif '```' in response:
            manim_code = response.split('```')[1].split('```')[0]
        else:
            manim_code = response
        fix_history = (
            f'You have a fix history which generates the code which is given to you:\n\n{fix_request}\n\n'
            f'If last error is same with latest error, **You probably find a wrong root cause**, '
            f'Check carefully and fix it again.**')
        return manim_code, fix_history

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
    def scale_video_to_fit(video_path, target_size=None):
        if target_size is None:
            target_size = RenderManim.window_size
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
