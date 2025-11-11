import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import List

from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.llm.openai_llm import OpenAI
from ms_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger(__name__)


@dataclass
class Pattern:

    name: str
    pattern: str
    tags: List[str] = field(default_factory=list)


class GenerateIllustrationPrompts(CodeAgent):

    line_art_prompt = """You is a scene description expert for AI knowledge science stickman videos. Based on the given knowledge point or storyboard, generate a detailed English description for a minimalist black-and-white stickman illustration with an AI/technology theme. Requirements:
- The illustration must depict only ONE scene, not multiple scenes, not comic panels, not split images. Absolutely do NOT use any comic panels, split frames, multiple windows, or any kind of visual separation. Each image is a single, unified scene.
- All elements (stickmen, objects, icons, patterns, tech elements, decorations) must appear together in the same space, on the same pure white background, with no borders, no frames, and no visual separation.
- All icons, patterns, and objects are decorative elements floating around or near the stickman, not separate scenes or frames. For example, do NOT draw any boxes, lines, or frames that separate parts of the image. All elements must be together in one open space.
- The background must be pure white. Do not describe any darkness, shadow, dim, black, gray, or colored background. Only describe a pure white background.
- All elements (stickmen, objects, tech elements, decorations) must be either solid black fill or outlined in black, to facilitate cutout. No color, no gray, no gradients, no shadows.
- The number of stickman characters should be chosen based on the meaning of the sentence: if the scene is suitable for a single person, use only one stickman; if it is suitable for interaction, use two or three stickmen. Do not force two or more people in every scene.
- All stickman characters must be shown as FULL BODY, with solid black fill for both body and face.
- Each stickman has a solid black face, with white eyes and a white mouth, both drawn as white lines. Eyes and mouth should be irregular shapes to express different emotions, not just simple circles or lines. Use these white lines to show rich, varied, and vivid emotions.
- Do NOT include any speech bubbles, text bubbles, comic panels, split images, or multiple scenes.
- All characters and elements must be fully visible, not cut off or overlapped.
"- Only add clear, readable English text in the image if it is truly needed to express the knowledge point or scene meaning, such as AI, Token, LLM, or any other relevant English word. Do NOT force the use of any specific word in every scene. If no text is needed, do not include any text. "
- All text in the image must be clear, readable, and not distorted, garbled, or random.
- Scene can include rich, relevant, and layered minimalist tech/AI/futuristic elements (e.g., computer, chip, data stream, AI icon, screen, etc.), and simple decorative elements to enhance atmosphere, but do not let elements overlap or crowd together.
- All elements should be relevant to the main theme and the meaning of the current subtitle segment.
- Output 80-120 words in English, only the scene description, no style keywords, and only use English text in the image if it is truly needed for the scene. """  # noqa

    color_prompt = """You are a scene description expert for AI knowledge science videos. Based on the given knowledge point or storyboard, generate a detailed English description for creating an appropriately styled illustration with an AI/technology theme. Requirements:

- The illustration must depict only ONE scene, not multiple scenes, not comic panels, not split images. Absolutely do NOT use any comic panels, split frames, multiple windows, or any kind of visual separation. Each image is a single, unified scene.
- All elements must appear together in the same space, with no borders, no frames, and no visual separation.
- All characters and elements must be fully visible, not cut off or overlapped.
- Only add clear, readable English text in the image if it is truly needed to express the knowledge point or scene meaning, such as AI, Token, LLM, or any other relevant English word. Do NOT force the use of any specific word in every scene. If no text is needed, do not include any text.
- All text in the image must be clear, readable, and not distorted, garbled, or random.
- The scene can include rich, relevant, and layered minimalist tech/AI/futuristic elements (e.g., computer, chip, data stream, AI icon, screen, etc.), and simple decorative elements to enhance atmosphere, but do not let elements overlap or crowd together.
- All elements should be relevant to the main theme and the meaning of the current subtitle segment.
- You must use the specified style, for example, 'comic', 'realistic', 'line-art'
- The image output should be a square, and its background should be **pure white**
- Image content should be uncluttered, with clear individual elements
- Unless necessary, do not generate text, as text may be generated incorrectly, creating an AI-generated feel
- The image panel size is 1920*1080, so you need to concentrate elements within a relatively flat image area. Elements at the top and bottom will be cropped
- The images need to accurately convey the meaning expressed by the text. Later, these images will be combined with text to create educational/knowledge-based videos
- Output 80-120 words in English, only the scene description, no style keywords, and only use English text in the image if it is truly needed for the scene."""  # noqa

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.llm: OpenAI = LLM.from_config(self.config)
        self.style = getattr(self.config.text2image, 't2i_style', 'realistic')
        self.system = self.line_art_prompt if self.style == 'line-art' else self.color_prompt
        self.illustration_prompts_dir = os.path.join(self.work_dir, 'illustration_prompts')
        os.makedirs(self.illustration_prompts_dir, exist_ok=True)

    async def execute_code(self, messages, **kwargs):
        with open(os.path.join(self.work_dir, 'segments.txt'), 'r') as f:
            segments = json.load(f)
        logger.info(f'Generating illustration prompts.')
        illustration_prompts = await asyncio.gather(*[
            self.generate_illustration_prompts(segment, i)
            for i, segment in enumerate(segments)
        ])
        assert len(illustration_prompts) == len(segments)
        for i, prompt in enumerate(illustration_prompts):
            with open(os.path.join(self.illustration_prompts_dir, f'segment_{i+1}.txt'), 'w') as f:
                f.write(prompt)
        return messages

    async def generate_illustration_prompts(self, segment, i):
        if os.path.exists(os.path.join(self.illustration_prompts_dir, f'segment_{i+1}.txt')):
            with open(os.path.join(self.illustration_prompts_dir, f'segment_{i+1}.txt'), 'r') as f:
                return f.read()
        line_art_query = (
            f'Please generate a detailed English scene description for an AI knowledge science stickman '
            f'illustration based on: {segment["content"]}\nRemember: The illustration must depict only ONE scene, '
            f'not multiple scenes, not comic panels, not split images. Absolutely do NOT use any comic panels, '
            f'split frames, multiple windows, or any kind of visual separation. '
            f'All elements must be solid black or outlined in black, and all faces must use irregular '
            f'white lines for eyes and mouth to express emotion. All elements should be relevant to the '
            f'main theme and the meaning of the current subtitle segment. All icons, patterns, and objects '
            f'are decorative elements floating around or near the stick man, not separate scenes or frames. '
            f'For example, do NOT draw any boxes, lines, or frames that separate parts of the image. '
            f'All elements must be together in one open space.')
        background = segment['background']
        manim_query = ''
        if segment.get('manim'):
            manim_query = (f'There is a manim animation at the front of the generated image: {segment["manim"]}, '
                           f'you need to make the image background not steal the focus from the manim animation.')
        colorful_query = (f'The style required from user is: {self.style}, '
                          f'illustration based on: {segment["content"]}, '
                          f'{manim_query}, '
                          f'Requirements from the storyboard designer: {background}')
        prompt = line_art_query if self.style == 'line-art' else colorful_query
        logger.info(f'Generating illustration prompt for : {segment["content"]}.')
        inputs = [
            Message(role='system', content=self.system),
            Message(role='user', content=prompt),
        ]
        _response_message = self.llm.generate(inputs)
        response = _response_message.content
        return response.strip()
