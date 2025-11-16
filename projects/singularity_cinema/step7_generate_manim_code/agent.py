# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union

import json
from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.utils import get_logger
from omegaconf import DictConfig
from PIL import Image

logger = get_logger()


class GenerateManimCode(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.num_parallel = getattr(self.config, 'llm_num_parallel', 10)
        self.images_dir = os.path.join(self.work_dir, 'images')
        self.manim_code_dir = os.path.join(self.work_dir, 'manim_code')
        os.makedirs(self.manim_code_dir, exist_ok=True)

    async def execute_code(self, messages: Union[str, List[Message]],
                           **kwargs) -> List[Message]:
        with open(os.path.join(self.work_dir, 'segments.txt'), 'r') as f:
            segments = json.load(f)
        with open(os.path.join(self.work_dir, 'audio_info.txt'), 'r') as f:
            audio_infos = json.load(f)
        logger.info('Generating manim code.')

        tasks = []
        for i, (segment, audio_info) in enumerate(zip(segments, audio_infos)):
            manim_requirement = segment.get('manim')
            if manim_requirement is not None:
                tasks.append((segment, audio_info['audio_duration'], i))

        manim_code = [''] * len(segments)

        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = {
                executor.submit(self._generate_manim_code_static, seg, dur,
                                idx, self.config, self.images_dir): idx
                for seg, dur, idx in tasks
            }
            for future in as_completed(futures):
                idx = futures[future]
                manim_code[idx] = future.result()

        for i, code in enumerate(manim_code):
            manim_file = os.path.join(self.manim_code_dir,
                                      f'segment_{i + 1}.py')
            with open(manim_file, 'w') as f:
                f.write(code)
        return messages

    @staticmethod
    def _generate_manim_code_static(segment, audio_duration, i, config,
                                    image_dir):
        """Static method for multiprocessing"""
        llm = LLM.from_config(config)
        return GenerateManimCode._generate_manim_impl(llm, segment,
                                                      audio_duration, i,
                                                      image_dir)

    @staticmethod
    def get_image_size(filename):
        with Image.open(filename) as img:
            return f'{img.width}x{img.height}'

    @staticmethod
    def get_all_images_info(segment, i, image_dir):
        all_images_info = []
        foreground = segment.get('foreground', [])
        for idx, _req in enumerate(foreground):
            foreground_image = os.path.join(
                image_dir, f'illustration_{i + 1}_foreground_{idx + 1}.png')
            size = GenerateManimCode.get_image_size(foreground_image)
            image_info = {
                'filename': foreground_image,
                'size': size,
                'description': _req,
            }
            all_images_info.append(image_info)

        image_info_file = os.path.join(
            os.path.dirname(image_dir), 'image_info.txt')
        if os.path.exists(image_info_file):
            with open(image_info_file, 'r') as f:
                for line in f.readlines():
                    if not line.strip():
                        continue
                    image_info = json.loads(line)
                    if image_info['filename'] in segment.get('user_image', []):
                        all_images_info.append(image_info)
        return all_images_info

    @staticmethod
    def _generate_manim_impl(llm, segment, audio_duration, i, image_dir):
        class_name = f'Scene{i + 1}'
        content = segment['content']
        manim_requirement = segment['manim']
        images_info = GenerateManimCode.get_all_images_info(
            segment, i, image_dir)
        if images_info:
            images_info = json.dumps(images_info, indent=4, ensure_ascii=False)
        else:
            images_info = 'No images offered.'

        prompt = f"""You are a professional Manim animation expert, creating clear and beautiful educational animations.

**Task**: Create animation
- Class name: {class_name}
- Content: {content}
- Requirement from the storyboard designer: {manim_requirement}
    * If the storyboard designer's layout is poor, create a better custom layout
- Duration: {audio_duration} seconds
- Code language: **Python**

**Image usage**
- You'll receive an actual image list with three fields per image: filename, size, and description，consider deeply how to use them in your animation
- Pay attention to the size field, write Manim code that respects the image's aspect ratio, size it if it's too big
- Consider the image integration with the background and overall animation. Use blending/glow effects, frames, movements, borders etc. to make it more beautiful and gorgeous
    * You can more freely consider the integration of images to achieve a better presentation
    * Images size should be medium or small to prevent them from occupying the entire screen or most of the screen, big image is not cool
    * Consider shape matching between images and surrounding components. circular to circular, square to square. Ensure aspect ratios of non-square images remain correct
    * If using any image, decorate it with a gorgeous frame
- [IMPORTANT] If images files is not empty, **you must use them all at the appropriate time and position in your animation**. Here is the image files list:

{images_info}

• Canvas size: (1250, 700) (width x height) which is the top 3/4 of screen, bottom is left for subtitles
• Ensure all content stays within safe bounds x∈(-6.0, 6.0), y∈(-3.4, 3.4) with minimum buff=0.5 from any edge to prevent cropping.
• [CRITICAL]Absolutely prevent **element spatial overlap** or **elements going out of bounds** or **elements not aligned**.
• [CRITICAL]Connection lines between boxes/text are of proper length, with **both endpoints attached to the objects**.
• All boxes must have thick strokes for clear visibility
• Keep text within frame by controlling font sizes. Use smaller fonts for Latin script than Chinese due to longer length.
• Ensure all pie chart pieces share the same center coordinates. Previous pie charts were drawn incorrectly.
• Use less stick man unless the user wants to, to prevent the animation from being too naive, try to make your effects more gorgeous/spectacular
• Concise and smooth animation effects
• Progressive display, avoid information overload

**Color Suggestions**:
• You need to explicitly specify element colors and make these colors coordinated and elegant in style.
• Consider the advices from the storyboard designer.
• **Don't use light yellow, light blue, grey**, etc., as this will make the animation look superficial, Consider white, black, dark blue, dark purple.

Please create Manim animation code that meets the above requirements.""" # noqa

        logger.info(f'Generating manim code for: {content}')
        _response_message = llm.generate(
            [Message(role='user', content=prompt)], temperature=0.3)
        response = _response_message.content
        if '```python' in response:
            manim_code = response.split('```python')[1].split('```')[0]
        elif '```' in response:
            manim_code = response.split('```')[1].split('```')[0]
        else:
            manim_code = response
        return manim_code
