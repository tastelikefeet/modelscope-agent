# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
from typing import List

import json
from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.llm.openai_llm import OpenAI
from ms_agent.utils import get_logger
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

logger = get_logger(__name__)


class GenerateSubtitle(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.llm: OpenAI = LLM.from_config(self.config)
        self.subtitle_lang = getattr(self.config, 'subtitle_lang', 'Chinese')
        self.subtitle_dir = os.path.join(self.work_dir, 'subtitles')
        os.makedirs(self.subtitle_dir, exist_ok=True)
        self.fonts = self.config.fonts

    async def execute_code(self, messages, **kwargs):
        with open(os.path.join(self.work_dir, 'segments.txt'), 'r') as f:
            segments = json.load(f)
        logger.info('Generating subtitles.')
        for i, seg in enumerate(segments):
            text = seg.get('content', '')
            subtitle = None
            if self.subtitle_lang:
                subtitle = await self.translate_text(text, self.subtitle_lang)
            output_file = os.path.join(self.subtitle_dir,
                                       f'bilingual_subtitle_{i + 1}.png')
            if os.path.exists(output_file):
                continue
            self.create_bilingual_subtitle_image(
                source=text,
                target=subtitle,
                output_file=output_file,
                width=1720,
                height=180)
        return messages

    @staticmethod
    def _split_subtitles(text: str, max_chars: int = 30) -> List[str]:
        # Only split at sentence endings (.!?), NOT at commas
        sentences = re.split(r'([。！？.!?])', text)
        subs, cur = [], ''
        for s in sentences:
            if not s.strip():
                continue
            test = cur + s
            if len(test) <= max_chars:
                cur = test
            else:
                if cur:
                    subs.append(cur.strip())
                cur = s
        if cur.strip():
            subs.append(cur.strip())
        return subs

    async def translate_text(self, text, to_lang):

        prompt = f"""You are a professional translation expert specializing in accurately and fluently translating text into {to_lang}.

## Skills

- Upon receiving content, translate it accurately into {to_lang}, ensuring the translation maintains the original meaning, tone, and style.
- Fully consider the context and cultural connotations to make the {to_lang} expression both faithful to the original and in line with {to_lang} conventions.
- Do not generate multiple translations for the same sentence.
- Output must conform to {to_lang} grammar standards, with clear, fluent expression and good readability.
- Accurately convey all information from the original text, avoiding arbitrary additions or deletions.
- Only provide services related to {to_lang} translation.
- Output only the translation result without any explanations.

Now translate:
""" # noqa
        messages = [
            Message(role='system', content=prompt),
            Message(role='user', content=text),
        ]

        _response_message = self.llm.generate(messages)
        return _response_message.content

    def get_font(self, size):
        """Get font using system font manager, same as CreateBackground agent"""
        import matplotlib.font_manager as fm
        for font_name in self.fonts:
            try:
                font_path = fm.findfont(fm.FontProperties(family=font_name))
                return ImageFont.truetype(font_path, size)
            except (OSError, ValueError):
                continue
        return ImageFont.load_default()

    def smart_wrap_text(self, text, max_lines=2, chars_per_line=50):
        import string

        # Only break at sentence endings (.!?), NOT commas
        sentence_enders = '.!?。！？'
        all_punctuation = string.punctuation + '。！？；，、：""' '《》【】（）'

        def is_chinese(char):
            return '\u4e00' <= char <= '\u9fff'

        def find_best_break(text, max_pos):
            if max_pos >= len(text):
                return len(text), False
            
            # Priority 1: Sentence endings (.!?)
            for i in range(max_pos, max(0, max_pos - 30), -1):
                if i > 0 and text[i-1] in sentence_enders:
                    return i, False
            
            # Priority 2: Whitespace
            for i in range(max_pos, max(0, max_pos - 20), -1):
                if text[i].isspace():
                    return i, False
            
            # Priority 3: Chinese characters
            if is_chinese(text[max_pos-1] if max_pos > 0 else text[0]):
                return max_pos, False
            
            return max_pos, False

        lines = []
        pos = 0
        while pos < len(text):
            remaining = text[pos:]
            if len(remaining) <= chars_per_line:
                lines.append(remaining.strip())
                break
            
            break_pos, _ = find_best_break(remaining, chars_per_line)
            line = remaining[:break_pos].strip()
            if line:
                lines.append(line)
            pos += break_pos
            while pos < len(text) and text[pos].isspace():
                pos += 1
        
        return lines[:max_lines] if lines else [text]

    def create_subtitle_image(self,
                              text,
                              width=1720,
                              height=120,
                              font_size=28,
                              text_color='black',
                              bg_color='rgba(0,0,0,0)',
                              chars_per_line=50):
        font = self.get_font(font_size)
        min_font_size = 18
        max_height = 500
        original_font_size = font_size
        lines = []
        while font_size >= min_font_size:
            if font_size != original_font_size:
                font = self.get_font(font_size)
            lines = self.smart_wrap_text(
                text, max_lines=2, chars_per_line=chars_per_line)
            line_height = font_size + 8
            total_text_height = len(lines) * line_height

            all_lines_fit = True
            for line in lines:
                bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox(
                    (0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                if line_width > width * 0.95:
                    all_lines_fit = False
                    break

            if total_text_height <= height and all_lines_fit:
                break
            elif total_text_height <= max_height and all_lines_fit:
                break
            else:
                font_size = int(font_size * 0.9)

        line_height = font_size + 8
        total_text_height = len(lines) * line_height
        actual_height = total_text_height + 16
        img = Image.new('RGBA', (width, actual_height), bg_color)
        draw = ImageDraw.Draw(img)
        y_start = 8
        for i, line in enumerate(lines):
            if not line.strip():
                continue

            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = max(0, (width - text_width) // 2)
            y = y_start + i * line_height

            if y + line_height <= actual_height and x >= 0 and x + text_width <= width:
                draw.text((x, y), line, fill=text_color, font=font)
        return img, actual_height

    def create_bilingual_subtitle_image(self,
                                        source,
                                        output_file,
                                        target='',
                                        width=1720,
                                        height=180):
        main_font_size = 32
        target_font_size = 22
        main_target_gap = 6
        chars_per_line = 50

        main_img, main_height = self.create_subtitle_image(
            source,
            width,
            height,
            main_font_size,
            'black',
            chars_per_line=chars_per_line)

        if target and target.strip():
            # For English, allow more characters per line due to narrower chars
            target_chars_per_line = 100
            target_img, target_height = self.create_subtitle_image(
                target,
                width,
                height,
                target_font_size,
                '#404040',  # Darker gray for better visibility
                chars_per_line=target_chars_per_line)
            total_height = main_height + target_height + main_target_gap
            combined_img = Image.new('RGBA', (width, total_height),
                                     (0, 0, 0, 0))
            combined_img.paste(main_img, (0, 0), main_img)
            combined_img.paste(target_img, (0, main_height + main_target_gap),
                               target_img)
            final_img = combined_img
            final_height = total_height
        else:
            final_img = main_img
            final_height = main_height

        final_img.save(output_file)
        return final_height
