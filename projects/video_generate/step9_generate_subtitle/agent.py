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
                        height=120)
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
                    height=120)
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
    def smart_wrap_text(text, font, max_width, max_lines=2):
        lines = []

        sample_char_width = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox(
            (0, 0), '中', font=font)[2]
        chars_per_line = int((max_width * 0.9) // sample_char_width)
        total_capacity = chars_per_line * max_lines
        if len(text) > total_capacity:
            truncate_pos = total_capacity - 3
            punctuation = [
                '。', '！', '？', '；', '，', '、', '.', '!', '?', ';', ','
            ]
            best_cut = truncate_pos

            for i in range(
                    min(len(text), truncate_pos), max(0, truncate_pos - 20),
                    -1):
                if text[i] in punctuation:
                    best_cut = i + 1
                    break
            text = text[:best_cut]

        sentences = re.split(r'([。！？；，、.!?;,])', text)
        current_line = ''
        for part in sentences:
            if not part.strip():
                continue

            test_line = current_line + part
            bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0),
                                                                     test_line,
                                                                     font=font)
            line_width = bbox[2] - bbox[0]
            if line_width <= max_width * 0.9 and len(lines) < max_lines:
                current_line = test_line
            else:
                if current_line.strip() and len(lines) < max_lines:
                    lines.append(current_line.strip())
                    current_line = part
                elif len(lines) >= max_lines:
                    break
        if current_line.strip() and len(lines) < max_lines:
            lines.append(current_line.strip())
        final_lines = []
        for line in lines:
            if len(final_lines) >= max_lines:
                break

            bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0),
                                                                     line,
                                                                     font=font)
            line_width = bbox[2] - bbox[0]
            if line_width <= max_width * 0.9:
                final_lines.append(line)
            else:
                chars = list(line)
                temp_line = ''
                for char in chars:
                    if len(final_lines) >= max_lines:
                        break

                    test_line = temp_line + char
                    bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox(
                        (0, 0), test_line, font=font)
                    test_width = bbox[2] - bbox[0]

                    if test_width <= max_width * 0.9:
                        temp_line = test_line
                    else:
                        if temp_line and len(final_lines) < max_lines:
                            final_lines.append(temp_line)
                        temp_line = char

                if temp_line and len(final_lines) < max_lines:
                    final_lines.append(temp_line)

        return final_lines[:max_lines]

    @staticmethod
    def create_subtitle_image(text,
                              width=1720,
                              height=120,
                              font_size=28,
                              text_color='black',
                              bg_color='rgba(0,0,0,0)'):
        font = GenerateSubtitle.load_font(font_size)
        min_font_size = 18
        max_height = 400
        original_font_size = font_size
        lines = []
        while font_size >= min_font_size:
            if font_size != original_font_size:
                font = GenerateSubtitle.load_font(font_size)
            lines = GenerateSubtitle.smart_wrap_text(
                text, font, width, max_lines=2)
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
                                        height=120):
        zh_font_size = 32
        en_font_size = 22
        zh_en_gap = 6
        zh_img, zh_height = GenerateSubtitle.create_subtitle_image(
            source, width, height, zh_font_size, 'black')

        if target.strip():
            en_img, en_height = GenerateSubtitle.create_subtitle_image(
                target, width, height, en_font_size, 'gray')
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