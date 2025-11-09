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
            if seg.get('type') != 'text':
                text = seg.get('content', '')
                parts = self._split_subtitles(text, max_chars=30)
                img_list = []
                for idx_p, part in enumerate(parts):
                    subtitle = await self.translate_text(
                        part, self.subtitle_lang)
                    output_file = os.path.join(
                        subtitle_dir,
                        f'bilingual_subtitle_{i + 1}_{idx_p + 1}.png')
                    self.create_bilingual_subtitle_image(
                        source=part,
                        target=subtitle,
                        output_file=output_file,
                        width=1720,
                        height=180)
                    img_list.append(output_file)
                context['subtitle_segments_list'].append(img_list)
                context['subtitle_paths'].append(
                    img_list[0] if img_list else None)
            else:
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
            font = ImageFont.truetype('msyh.ttc', font_size)
        except OSError or ValueError:
            try:
                font = ImageFont.truetype('arial.ttf', font_size)
            except OSError or ValueError:
                font = ImageFont.load_default()
        return font

    @staticmethod
    def smart_wrap_text(text, font, max_width, max_lines=2, chars_per_line=15):
        """Smart text wrapping with character-based line breaks.
        
        Args:
            text: Text to wrap
            font: PIL ImageFont object
            max_width: Maximum width in pixels
            max_lines: Maximum number of lines
            chars_per_line: Target characters per line (default 15)
        """
        lines = []
        
        # Split text into chunks of ~15 characters, respecting punctuation
        punctuation = ['。', '！', '？', '；', '，', '、', '.', '!', '?', ';', ',']
        
        current_line = ''
        char_count = 0
        
        for i, char in enumerate(text):
            current_line += char
            char_count += 1
            
            # Check if we should break the line
            should_break = False
            
            # Break at punctuation near target length
            if char in punctuation and char_count >= chars_per_line - 3:
                should_break = True
            # Force break at target length if no punctuation found
            elif char_count >= chars_per_line:
                should_break = True
            # Break at punctuation if line is getting long
            elif char in punctuation and char_count >= chars_per_line * 0.8:
                should_break = True
            
            # Also verify pixel width
            if should_break or i == len(text) - 1:
                bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox(
                    (0, 0), current_line, font=font)
                line_width = bbox[2] - bbox[0]
                
                # If line is too wide, force break earlier
                if line_width > max_width * 0.95 and len(current_line) > 1:
                    # Try to break at last punctuation
                    for j in range(len(current_line) - 1, 0, -1):
                        if current_line[j] in punctuation:
                            lines.append(current_line[:j+1].strip())
                            current_line = current_line[j+1:]
                            char_count = len(current_line)
                            break
                    else:
                        # No punctuation found, hard break
                        lines.append(current_line[:-1].strip())
                        current_line = current_line[-1]
                        char_count = 1
                elif should_break:
                    lines.append(current_line.strip())
                    current_line = ''
                    char_count = 0
                
                if len(lines) >= max_lines:
                    break
        
        # Add remaining text
        if current_line.strip() and len(lines) < max_lines:
            lines.append(current_line.strip())
        
        return lines[:max_lines]

    @staticmethod
    def create_subtitle_image(text,
                              width=1720,
                              height=120,
                              font_size=28,
                              text_color='black',
                              bg_color='rgba(0,0,0,0)',
                              chars_per_line=15):
        font = GenerateSubtitle.load_font(font_size)
        min_font_size = 28  # Increased from 24 to maintain larger minimum size
        max_height = 500
        original_font_size = font_size
        lines = []
        while font_size >= min_font_size:
            if font_size != original_font_size:
                font = GenerateSubtitle.load_font(font_size)
            lines = GenerateSubtitle.smart_wrap_text(
                text, font, width, max_lines=2, chars_per_line=chars_per_line)
            line_height = font_size + 15  # Increased spacing from 8 to 15
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

        line_height = font_size + 15  # Increased spacing
        total_text_height = len(lines) * line_height
        actual_height = total_text_height + 20  # Increased padding from 16 to 20
        img = Image.new('RGBA', (width, actual_height), bg_color)
        draw = ImageDraw.Draw(img)
        y_start = 10  # Increased from 8 to 10
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
        # Significantly increased font sizes for better readability
        zh_font_size = 68  # Increased from 48 to 68
        en_font_size = 44  # Increased from 32 to 44
        zh_en_gap = 12  # Increased gap from 10 to 12
        chars_per_line = 15  # Target 15 characters per line
        
        zh_img, zh_height = GenerateSubtitle.create_subtitle_image(
            source, width, height, zh_font_size, 'black', chars_per_line=chars_per_line)

        if target.strip():
            # For English, allow more characters per line due to narrower chars
            en_chars_per_line = 25
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