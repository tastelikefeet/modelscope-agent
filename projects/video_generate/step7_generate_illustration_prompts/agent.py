import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import List

from omegaconf import DictConfig

from ms_agent.agent import CodeAgent
from ms_agent.llm import Message, LLM
from ms_agent.llm.openai_llm import OpenAI
from ms_agent.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Pattern:

    name: str
    pattern: str
    tags: List[str] = field(default_factory=list)


class GenerateIllustrationPrompts(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.animation_mode = os.environ.get('MS_ANIMATION_MODE',
                                              'auto').strip().lower() or 'auto'
        self.llm: OpenAI = LLM.from_config(self.config)

    async def run(self, inputs, **kwargs):
        messages, context = inputs
        segments = context['segments']
        text_segments = [
            seg for seg in segments if seg.get('type') == 'text'
        ]
        illustration_prompts = await asyncio.gather(*[
            self.generate_illustration_prompts(segment)
            for segment in text_segments
        ])
        context['illustration_prompts'] = illustration_prompts

    async def generate_illustration_prompts(self, segment):
        system_prompt = """You is a scene description expert for AI knowledge science stickman videos. Based on the given knowledge point or storyboard, generate a detailed English description for a minimalist black-and-white stickman illustration with an AI/technology theme. Requirements:
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

        prompt = (
            f'Please generate a detailed English scene description for an AI knowledge science stickman '
            f'illustration based on: {segment}\nRemember: The illustration must depict only ONE scene, '
            f'not multiple scenes, not comic panels, not split images. Absolutely do NOT use any comic panels, '
            f'split frames, multiple windows, or any kind of visual separation. '
            f'All elements must be solid black or outlined in black, and all faces must use irregular '
            f'white lines for eyes and mouth to express emotion. All elements should be relevant to the '
            f'main theme and the meaning of the current subtitle segment. All icons, patterns, and objects '
            f'are decorative elements floating around or near the stickman, not separate scenes or frames. '
            f'For example, do NOT draw any boxes, lines, or frames that separate parts of the image. '
            f'All elements must be together in one open space.')

        inputs = [
            Message(role='system', content=system_prompt),
            Message(role='user', content=prompt),
        ]
        _response_message = self.llm.generate(inputs)
        response = _response_message.content
        return response.strip()
