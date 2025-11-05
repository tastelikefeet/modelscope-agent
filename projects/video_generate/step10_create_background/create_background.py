import os
from typing import List, Union

import json
from ms_agent.agent.base import Agent
from ms_agent.llm import Message
from omegaconf import DictConfig
from projects.video_generate.core import workflow as video_workflow


class CreateBackground(Agent):
    """A thin wrapper that dispatches to original workflow functions.
    It preserves all original prompts/logic. We only adapt to ms-agent's CodeAgent loading via code_file.
    """

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        # work_dir for intermediates
        self.work_dir = os.path.join(
            self.config.local_dir, 'output') if getattr(
                self.config, 'local_dir', None) else os.getcwd()
        os.makedirs(self.work_dir, exist_ok=True)
        self.meta_path = os.path.join(self.work_dir, 'meta.json')
        # animation mode: auto (default) or human (manual animation workflow)
        import os as _os
        self.animation_mode = _os.environ.get('MS_ANIMATION_MODE',
                                              'auto').strip().lower() or 'auto'
        print(f'[video_agent] Animation mode: {self.animation_mode}')

    def create_manual_background(title_text='', output_dir='output', topic=None):
        """默认背景样式"""

        from PIL import Image, ImageDraw, ImageFont
        import os
        import textwrap

        os.makedirs(output_dir, exist_ok=True)
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

        def _get_font(size):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            import matplotlib.font_manager as fm
            font_names = [
                'SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Microsoft YaHei'
            ]
            # 首先尝试加载本地字体文件
            local_font = os.path.join(script_dir, 'asset', '字小魂扶摇手书(商用需授权).ttf')
            try:
                return ImageFont.truetype(local_font, size)
            except Exception as e:
                print(f'本地字体加载失败: {local_font}, 错误: {str(e)}')
            # 尝试使用matplotlib查找系统中的中文字体
            for font_name in font_names:
                try:
                    font_path = fm.findfont(fm.FontProperties(family=font_name))
                    return ImageFont.truetype(font_path, size)
                except Exception as e:
                    print(f'无法找到字体: {font_name}, 错误: {str(e)}')
                    continue

            print('所有字体加载失败，使用默认字体')
            return ImageFont.load_default()

        title_font = _get_font(config['title_font_size'])
        subtitle_font = _get_font(config['subtitle_font_size'])

        title_display = title_text or 'AI知识科普'
        title_lines = textwrap.wrap(title_display, width=config['title_max_width'])
        y_position = config['padding']
        for line in title_lines:
            bbox = draw.textbbox((0, 0), line, font=title_font)
            draw.text((config['padding'], y_position),
                      line,
                      font=title_font,
                      fill=title_color)
            y_position += (bbox[3] - bbox[1]) + config['line_spacing']
        subtitle_lines = ['硬核知识分享', '魔搭社区出品']
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

        if topic:
            # 清理topic中的特殊字符，避免路径问题
            import re
            safe_topic = re.sub(r'[^\w\u4e00-\u9fff\-_]', '_',
                                topic)  # 只保留字母、数字、中文、横线、下划线
            safe_topic = safe_topic[:50]  # 限制长度
            theme_dir = os.path.join(output_dir, safe_topic)
            os.makedirs(theme_dir, exist_ok=True)
            output_path = os.path.join(theme_dir, f'background_{uuid.uuid4()}.png')
        else:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir,
                                       f'background_{uuid.uuid4()}.png')
        image.save(output_path)
        print(f'使用统一背景样式生成: {output_path}')
        return output_path

    async def run(self, inputs: Union[str, List[Message]],
                  **kwargs) -> List[Message]:

