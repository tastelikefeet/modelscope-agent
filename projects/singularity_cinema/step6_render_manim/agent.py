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
                output_text = RenderManim.check_manim_quality(final_file_path, work_dir, i, config, segment, cur_check_round)
                # output_text = RenderManim.generate_fix_prompts(llm, output_text, code, segment)
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
            RenderManim._extract_preview_frames_static(final_file_path, i, work_dir, 'final')
            # manim_code_dir = os.path.join(work_dir, 'manim_code')
            # manim_file = os.path.join(manim_code_dir, f'segment_{i + 1}.py')
            # with open(manim_file, 'w') as f:
            #     f.write(code)
        else:
            raise FileNotFoundError(final_file_path)

    @staticmethod
    def generate_fix_prompts(llm, output_text, code, segment):
        system = """You are an assistant responsible for helping resolve multi-modal LLM feedback issues in short video generation. Your role is to identify whether these issues exist and generate the correct code fix prompts.

You are an assistant responsible for helping resolve human feedback issues in short video generation. Your role is to identify which workflow step the reported problem occurs in based on human feedback, and appropriately delete configuration files of prerequisite tasks to trigger task re-execution.

Workflow Overview:
First, there is a root directory folder for storing all files. All files described below and all your tool commands are based on this root directory. You don't need to worry about the root directory location; just focus on relative directories.

Steps related to you:

- Generate basic script based on user requirements

Output: script file script.txt, original requirements file topic.txt, video title file title.txt

- Segment design based on script

Output: segments.txt, describing a list of shots including narration, background image generation requirements, and foreground Manim animation requirements

- Generate audio narration for segments

- Generate Manim animation code based on audio duration

Output: list of Manim code files manim_code/segment_N.py, where N is the segment number starting from 1

- Render Manim code

Output: list of manim_render/scene_N folders. If segments.txt contains Manim requirements for a certain step, the corresponding folder will have a manim.mov file

- Generate text-to-image prompts, images, and other materials

- Compose final video

- Your work is in step 5. An MLLM is used to analyze the layout problems, but they are not accurate. You need to check the issues and code files and give your fix prompts as accurately as possible.

- The MLLM model may give incorrect feedbacks, you need to check the code and ignore problems that meet this condition, stand with your code if you insist you are right!

- You need to trust your code if you believe the issue is a false positive

Now begin:"""

        content = segment['content']
        manim_requirement = segment['manim']
        query = f"""Manim origin requirements:
- Content: {content}
- Requirement from the storyboard designer: {manim_requirement}
- Code language: **Python**

Feedbacks from the MLLM: {output_text}

Current Code: {code}

Wrap your fix prompts with <result>...</result>, if no need to fix, leave an empty content <result></result>.

Now generate your fix prompts:
"""
        inputs = [Message(role='system', content=system), Message(role='user', content=query)]
        response = llm.generate(inputs)
        pattern = r'<result>(.*?)</result>'
        issues = []
        for issue in re.findall(pattern, response.content, re.DOTALL):
            issues.append(issue)
        return '\n'.join(issues).strip()

    @staticmethod
    def check_manim_quality(final_file_path, work_dir, i, config, segment, cur_check_round):
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
        test_system = """
**角色定位**
你是Manim动画布局检查专家，负责检查动画帧中的布局问题。

**背景信息**
- 你收到的图片是Manim渲染的视频帧（中间帧或最终帧）
- 视频尺寸：1920×1080，可用渲染区域：1400×700

**检查重点**

**必须报告的严重问题：**
1. 组件或文本重叠
2. 组件或文本被边缘裁切（哪怕轻微裁切）
3. 父子组件不一致（子元素超出父容器边界）
4. 图表元素错位（饼图中心偏移、柱状图/折线图位置错误）

**需要报告的次要问题：**
1. 功能相同的组件未对齐
2. 连接线起点/终点错误、箭头方向错误、线条与组件重叠

**检查规则**
- 中间帧：只关注重叠问题，忽略不完整组件
- 最终帧：检查所有上述问题
- 忽略：美学问题、因动画过程导致的临时不合理位置

**输出格式**

```
<description>
详细描述图片内容，包括所有组件的位置及其与边缘、其他组件的距离
</description>

<result>
列出发现的问题及修复建议。如无问题则留空。
</result>
```

**示例：**
```
<description>
图中有四个方形组件，第一个组件距离左侧约...
</description>

<result>
右侧组件被挤压到边缘。修复建议：缩小左侧四个组件宽度，右移右侧组件...
</result>
```
"""

        test_images = RenderManim._extract_preview_frames_static(final_file_path, i, work_dir, cur_check_round)
        llm = LLM.from_config(_mm_config)

        frame_names = ['the middle frame of the animation', 'the last frame of the animation']
        content = segment['content']
        manim_requirement = segment['manim']

        all_issues = []
        for idx, (image_path, frame_name) in enumerate(zip(test_images, frame_names)):
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')

            _content = [
                {
                    "type": "text",
                    "text": (f"The current frame is: {frame_name}, the content of this animation: {content}, "
                            f"the manim animation requirement: {manim_requirement}, "
                            f"you must carefully check the animation layout issues.")
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
                Message(role='system', content=test_system),
                Message(role='user', content=_content),
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

            pattern = r'<description>(.*?)</description>'
            desc = []
            for _desc in re.findall(pattern, response_text, re.DOTALL):
                desc.append(_desc)
            desc = '\n'.join(desc).strip()
            if issues:
                issues = f'Current is the {frame_name}, problem checked by a MLLM: {issues}, frame description: {desc}'
            all_issues.append(issues)

        all_issues =  '\n'.join(all_issues).strip()
        return all_issues

    @staticmethod
    def _extract_preview_frames_static(video_path, segment_id, work_dir, cur_check_round):
        from moviepy import VideoFileClip
        
        test_dir = os.path.join(work_dir, 'manim_test')
        os.makedirs(test_dir, exist_ok=True)
        video = VideoFileClip(video_path)
        duration = video.duration

        timestamps = {
            1: duration / 2,
            2: max(0, duration - 0.5)
        }

        preview_paths = []
        for frame_idx, timestamp in timestamps.items():
            output_path = os.path.join(
                test_dir,
                f'segment_{segment_id + 1}_round{cur_check_round}_{frame_idx}.png'
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
- Duration: {audio_duration} seconds
- Code language: **Python**

Manim instructions:

**Spatial Constraints (CRITICAL)**:
• Canvas size: (1400, 700) (width x height) which is the top 3/4 of screen, bottom is left for subtitles
• Safe area: x∈(-6.5, 6.5), y∈(-3.3, 3.3) (0.5 units from edge)
• Element spacing: Use buff=0.3 or larger (avoid overlap)
• Relative positioning: Prioritize next_to(), align_to(), shift()
• Avoid multiple elements using the same reference point
• [CRITICAL]Absolutely prevent **element spatial overlap** or **elements going out of bounds** or **elements not aligned**.
• [CRITICAL]Connection lines between boxes/text are of proper length, with **both endpoints attached to the objects**.

**Box/Rectangle Size Standards**:
• For diagram boxes: Use consistent dimensions, e.g., Rectangle(width=2.5, height=1.5)
• For labels/text boxes: width=1.5~3.0, height=0.8~1.2
• For emphasis boxes: width=3.0~4.0, height=1.5~2.0
• Always specify both width AND height explicitly: Rectangle(width=2.5, height=1.5)
• Avoid using default sizes - always set explicit dimensions
• Maintain consistent box sizes within the same diagram level/category
• All boxes must have thick strokes for clear visibility
• Keep text within frame by controlling font sizes. Use smaller fonts for Latin script than Chinese due to longer length.
• Ensure all pie chart pieces share the same center coordinates. Previous pie charts were drawn incorrectly.

**Visual Quality Enhancement**:
• Use thick, clear strokes for all shapes
    - 4~6 strokes is recommended
• Make arrows bold and prominent
• Add rounded corners for modern aesthetics: RoundedRectangle(corner_radius=0.15)
• Use subtle fill colors with transparency when appropriate: fill_opacity=0.1
• Ensure high contrast between elements for clarity
• Apply consistent spacing and alignment throughout
• Use less stick man unless the user wants to, to prevent the animation from being too naive, try to make your effects more dazzling/gorgeous/spectacular/blingbling

**Layout Suggestions**:
• Content clearly layered
• Key information highlighted
• Reasonable use of space
• Maintain visual balance
• LLMs excel at animation complexity, not layout complexity. 
    - Use multiple storyboard scenes rather than adding more elements to one animation to avoid layout problems
    - For animations with many elements, consider layout carefully. For instance, arrange elements horizontally given the canvas's wider width
    - With four or more horizontal elements, put summary text or similar content at the canvas bottom, this will effectively reduce the cutting off and overlap problems

**Animation Requirements**:
• Concise and smooth animation effects
• Progressive display, avoid information overload
• Appropriate pauses and rhythm
• Professional visual presentation with thick, clear lines
• Use GrowArrow for arrows instead of Create for better effect
• Consider using Circumscribe or Indicate to highlight important elements

**Code Style**:
• Clear comments and explanations
• Avoid overly complex structures

**Color Suggestions**:
• You need to explicitly specify element colors and make these colors coordinated and elegant in style.
• Consider the advices from the storyboard designer.
• Don't use light yellow, light blue, etc., as this will make the animation look superficial.
• Consider more colors like white, black, dark blue, dark purple, dark orange, etc. DO NOT use grey color, it's not easy to read

- Please focus on solving the detected issues
- If you find other issues, fix them too
- Check any element does not match the instructions above
- Keep the good parts, do minimum changes, only fix problematic areas
- Ensure that the components do not overlap or get cut off by the edges
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
