import os
import textwrap

from ms_agent.agent.base import Agent
from ms_agent.llm import LLM
from ms_agent.llm.openai_llm import OpenAI
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont


class CreateBackground(Agent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.bg_path = os.path.join(self.work_dir, 'background.jpg')
        self.llm: OpenAI = LLM.from_config(self.config)
        self.fonts = getattr(
            self.config, 'fonts',
            ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Microsoft YaHei'])
        self.slogan = getattr(self.config, 'slogan', [])

    def get_font(self, size):
        import matplotlib.font_manager as fm
        local_font = os.path.join(
            os.path.dirname(__file__), '字小魂扶摇手书(商用需授权).ttf')
        try:
            return ImageFont.truetype(local_font, size)
        except OSError or ValueError:
            for font_name in self.fonts:
                try:
                    font_path = fm.findfont(
                        fm.FontProperties(family=font_name))
                    return ImageFont.truetype(font_path, size)
                except OSError or ValueError:
                    continue

        return ImageFont.load_default()

    async def run(self, inputs, **kwargs):
        messages, context = inputs
        topic = context.get('topic')
        width, height = 1920, 1080
        background_color = (255, 255, 255)
        title_color = (0, 0, 0)

        config = {
            'title_font_size': 50,
            'subtitle_font_size': 54,
            'title_max_width': 15,
            'subtitle_color': (0, 0, 0),
            'line_spacing': 15,
            'padding': 50,
            'line_width': 8,
            'subtitle_offset': 40,
            'line_position_offset': 190
        }

        image = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(image)

        title_font = self.get_font(config['title_font_size'])
        subtitle_font = self.get_font(config['subtitle_font_size'])

        title_lines = textwrap.wrap(topic, width=config['title_max_width'])
        y_position = config['padding']
        for line in title_lines:
            bbox = draw.textbbox((0, 0), line, font=title_font)
            draw.text((config['padding'], y_position),
                      line,
                      font=title_font,
                      fill=title_color)
            y_position += (bbox[3] - bbox[1]) + config['line_spacing']
        subtitle_lines = self.slogan
        y_position = config['padding']
        for i, line in enumerate(subtitle_lines):
            bbox = draw.textbbox((0, 0), line, font=subtitle_font)
            x_offset = width - bbox[2] - (config['padding'] + 30) + (
                i * config['subtitle_offset'])
            draw.text((x_offset, y_position),
                      line,
                      font=subtitle_font,
                      fill=config['subtitle_color'])
            y_position += bbox[3] - bbox[1] + 5

        line_y = height - config['padding'] - config['line_position_offset']
        draw.line([(0, line_y), (width, line_y)],
                  fill=(0, 0, 0),
                  width=config['line_width'])
        image.save(self.bg_path)
        context['background_path'] = self.bg_path
        return messages, context
