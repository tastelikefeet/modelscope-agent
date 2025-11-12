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
        self.subtitle_lang = getattr(self.config, 'subtitle_lang', 'English')
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
        sentences = re.split(r'([。！？；，、.!?;,])', text)
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

        # Define sentence-ending punctuation (highest priority for breaks)
        sentence_enders = '.!?。！？'
        # All punctuation marks
        all_punctuation = string.punctuation + '。！？；，、：""' '《》【】（）'

        def is_only_punctuation(s):
            """Check if string contains only punctuation and whitespace"""
            return all(c in all_punctuation or c.isspace() for c in s)

        def is_chinese(char):
            """Check if character is Chinese"""
            return '\u4e00' <= char <= '\u9fff'

        def find_best_break_point(text, max_pos):
            """Find the best position to break text, prioritizing sentence boundaries"""
            if max_pos >= len(text):
                return len(text), False

            # Priority 1: Look for sentence-ending punctuation within reasonable range
            # Search backward from max_pos, willing to go back up to 30% of line length
            search_start = max(0, int(max_pos * 0.7))
            for i in range(max_pos, search_start, -1):
                if i > 0 and i < len(text) and text[i - 1] in sentence_enders:
                    # Found sentence end, break after it
                    return i, False

            # Priority 2: Look for whitespace
            for i in range(max_pos, search_start, -1):
                if i < len(text) and text[i].isspace():
                    return i, False

            # Priority 3: Look for existing hyphens in compound words (e.g., "Megatron-LM")
            # Break after the hyphen to keep compound words together
            for i in range(max_pos, search_start, -1):
                if i > 0 and i < len(text) and text[i - 1] == '-':
                    return i, False

            # Priority 4: For Chinese text, can break between characters
            if max_pos > 0 and max_pos < len(text) and is_chinese(
                    text[max_pos - 1]):
                return max_pos, False

            # Priority 5: For English, try to break at word boundary
            # Look back for space
            for i in range(max_pos, max(0, max_pos - 15), -1):
                if i < len(text) and text[i].isspace():
                    return i, False

            # Last resort: force break with hyphen for English words
            if max_pos > 0 and max_pos < len(text):
                if (text[max_pos - 1].isalpha() and text[max_pos].isalpha()
                        and not is_chinese(text[max_pos - 1])
                        and not is_chinese(text[max_pos])):
                    return max_pos - 1, True  # True indicates we need to add hyphen

            return max_pos, False

        def break_line(text, max_chars):
            """Break text into lines with smart sentence-aware breaking"""
            if len(text) <= max_chars:
                return [text]

            lines = []
            current_pos = 0

            while current_pos < len(text):
                remaining = text[current_pos:]

                if len(remaining) <= max_chars:
                    lines.append(remaining)
                    break

                # Find best break point
                break_pos, needs_hyphen = find_best_break_point(
                    remaining, max_chars)

                if needs_hyphen:
                    # Add hyphen for word break
                    line = remaining[:break_pos] + '-'
                    lines.append(line)
                    current_pos += break_pos
                else:
                    # Extract the line
                    line = remaining[:break_pos].rstrip()
                    lines.append(line)
                    current_pos += break_pos

                    # Skip any leading whitespace for the next line
                    while current_pos < len(
                            text) and text[current_pos].isspace():
                        current_pos += 1

            return lines

        # Break text into initial lines
        raw_lines = break_line(text.strip(), chars_per_line)

        # Post-process lines
        processed_lines = []
        for line in raw_lines:
            # Strip leading/trailing whitespace
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip lines with only punctuation
            if is_only_punctuation(line):
                continue

            # Keep the line (don't strip ending punctuation - we want sentence enders)
            processed_lines.append(line)

        # Limit to max_lines
        if len(processed_lines) > max_lines:
            processed_lines = processed_lines[:max_lines]

        # If no valid lines, return original text as fallback
        if not processed_lines:
            processed_lines = [text.strip()]

        return processed_lines

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

        if target.strip():
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
