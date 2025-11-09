import os
import re
from typing import List

from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.llm.openai_llm import OpenAI
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont
from ms_agent.utils import get_logger

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

    async def execute_code(self, inputs, **kwargs):
        messages, context = inputs
        segments = context['segments']
        context['subtitle_segments_list'] = []
        context['subtitle_paths'] = []
        subtitle_dir = os.path.join(self.work_dir, 'subtitles')
        os.makedirs(subtitle_dir, exist_ok=True)
        logger.info(f'Generating subtitles.')
        for i, seg in enumerate(segments):
            text = seg.get('content', '')
            subtitle = await self.translate_text(text, self.subtitle_lang)
            output_file = os.path.join(subtitle_dir,
                                       f'bilingual_subtitle_{i + 1}.png')
            self.create_bilingual_subtitle_image(
                source=text,
                target=subtitle,
                output_file=output_file,
                width=1720,
                height=180)
            context['subtitle_segments_list'].append(output_file)
            context['subtitle_paths'].append([output_file])
        return messages, context

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

    @staticmethod
    def load_font(font_size):
        try:
            font = ImageFont.truetype('Alibaba-PuHuiTi-Medium.otf', font_size)
        except OSError or ValueError:
            try:
                font = ImageFont.truetype('arial.ttf', font_size)
            except OSError or ValueError:
                font = ImageFont.load_default(font_size)
        return font

    @staticmethod
    def smart_wrap_text(text, font, max_width, max_lines=2, chars_per_line=50):
        """Smart text wrapping with character-based line breaks.
        
        Args:
            text: Text to wrap
            font: PIL ImageFont object
            max_width: Maximum width in pixels
            max_lines: Maximum number of lines
            chars_per_line: Target characters per line (default 20)
        """
        import string
        
        # Define punctuation marks (both Chinese and English)
        punctuation = string.punctuation + '。！？；，、：""''《》【】（）'
        
        def is_only_punctuation(s):
            """Check if string contains only punctuation and whitespace"""
            return all(c in punctuation or c.isspace() for c in s)
        
        def strip_line_punctuation(s):
            """Remove punctuation from start and end of line"""
            punctuation = string.punctuation + '。！？；，、：""''《》【】（）'
            # Strip trailing punctuation
            while s and s[-1] in punctuation:
                s = s[:-1]
            return s
        
        def is_chinese(char):
            """Check if character is Chinese"""
            return '\u4e00' <= char <= '\u9fff'
        
        def can_break_here(text, pos):
            """Check if we can break at this position"""
            if pos >= len(text):
                return True
            # Can break at space
            if text[pos].isspace():
                return True
            # Can break after punctuation
            if pos > 0 and text[pos - 1] in punctuation:
                return True
            # Can break between Chinese characters
            if pos > 0 and is_chinese(text[pos - 1]):
                return True
            return False
        
        def break_line(text, max_chars):
            """Break text into lines with smart word breaking"""
            if len(text) <= max_chars:
                return [text]
            
            lines = []
            current_pos = 0
            
            while current_pos < len(text):
                remaining = text[current_pos:]
                
                if len(remaining) <= max_chars:
                    lines.append(remaining)
                    break
                
                # Try to find a good break point
                break_pos = max_chars
                found_break = False
                
                # Look backward for a natural break point
                for i in range(max_chars, max(0, max_chars - 10), -1):
                    if can_break_here(remaining, i):
                        break_pos = i
                        found_break = True
                        break
                
                # If no natural break found and we're in a word, add hyphen
                if not found_break and break_pos < len(remaining):
                    # Check if we're breaking in the middle of an English word
                    if (break_pos > 0 and 
                        remaining[break_pos - 1].isalpha() and 
                        break_pos < len(remaining) and 
                        remaining[break_pos].isalpha() and
                        not is_chinese(remaining[break_pos - 1]) and
                        not is_chinese(remaining[break_pos])):
                        # Add hyphen for word break
                        line = remaining[:break_pos - 1] + '-'
                        lines.append(line)
                        current_pos += break_pos - 1
                        continue
                
                # Extract the line
                line = remaining[:break_pos].rstrip()
                lines.append(line)
                current_pos += break_pos
                
                # Skip any leading whitespace for the next line
                while current_pos < len(text) and text[current_pos].isspace():
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
            
            # Remove punctuation from start and end
            cleaned_line = strip_line_punctuation(line)
            
            # Only add non-empty cleaned lines
            if cleaned_line.strip():
                processed_lines.append(cleaned_line.strip())
        
        # Limit to max_lines
        if len(processed_lines) > max_lines:
            processed_lines = processed_lines[:max_lines]
        
        # If no valid lines, return original text as fallback
        if not processed_lines:
            processed_lines = [text.strip()]
        
        return processed_lines

    @staticmethod
    def create_subtitle_image(text,
                              width=1720,
                              height=120,
                              font_size=28,
                              text_color='black',
                              bg_color='rgba(0,0,0,0)',
                              chars_per_line=50):
        font = GenerateSubtitle.load_font(font_size)
        min_font_size = 18
        max_height = 500
        original_font_size = font_size
        lines = []
        while font_size >= min_font_size:
            if font_size != original_font_size:
                font = GenerateSubtitle.load_font(font_size)
            lines = GenerateSubtitle.smart_wrap_text(
                text, font, width, max_lines=2, chars_per_line=chars_per_line)
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

    @staticmethod
    def create_bilingual_subtitle_image(source,
                                        output_file,
                                        target='',
                                        width=1720,
                                        height=180):
        zh_font_size = 32
        en_font_size = 22
        zh_en_gap = 6
        chars_per_line = 50
        
        zh_img, zh_height = GenerateSubtitle.create_subtitle_image(
            source, width, height, zh_font_size, 'black', chars_per_line=chars_per_line)

        if target.strip():
            # For English, allow more characters per line due to narrower chars
            en_chars_per_line = 100
            en_img, en_height = GenerateSubtitle.create_subtitle_image(
                target, width, height, en_font_size, 'gray', chars_per_line=en_chars_per_line)
            total_height = zh_height + en_height + zh_en_gap
            combined_img = Image.new('RGBA', (width, total_height),
                                     (0, 0, 0, 0))
            combined_img.paste(zh_img, (0, 0), zh_img)
            combined_img.paste(en_img, (0, zh_height + zh_en_gap), en_img)
            final_img = combined_img
            final_height = total_height
        else:
            final_img = zh_img
            final_height = zh_height

        final_img.save(output_file)
        return final_height

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