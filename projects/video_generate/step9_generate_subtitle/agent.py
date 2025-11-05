import os
from typing import List, Union

import json
from ms_agent.agent.base import Agent
from ms_agent.llm import Message
from omegaconf import DictConfig
from projects.video_generate.core import workflow as video_workflow


class GenerateSubtitle(Agent):
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

    def translate_text_to_english(text):
        """
        将中文文本翻译为英文
        """

        prompt = """

# 角色
你是一位专业的翻译专家，擅长将中文文本准确流畅地翻译成英文。

## 技能
- 接收到中文内容后，将其准确翻译成英文，确保译文保持原文的意义、语气和风格。
- 充分考虑中文的语境和文化内涵，使英文表达既忠实原文又符合英语习惯。
- 禁止同一句子生成多份译文。
- 输出内容需符合英语语法规范，表达清晰、流畅，并具有良好的可读性。
- 准确传达原文所有信息，避免随意添加或删减内容。
- 仅提供与中文到英文翻译相关的服务。
- 只输出翻译结果，不要任何说明。

"""

        try:
            print(f'[翻译DEBUG] 原文: {text[:50]}...')
            full_prompt = f'{prompt}\n原文：{text}\n译文：'
            print(f'[翻译DEBUG] 完整提示词: {full_prompt[:100]}...')
            result = modai_model_request(
                full_prompt,
                model='Qwen/Qwen3-Coder-480B-A35B-Instruct',
                max_tokens=512,
                temperature=0.3)
            print(f'[翻译DEBUG] 翻译结果: {result}')
            print(f'[翻译DEBUG] 结果类型: {type(result)}')
            return result.strip() if result else ''
        except Exception as e:
            print(f'英文翻译失败: {e}')
            return ''

    def create_bilingual_subtitle_image(zh_text,
                                        en_text='',
                                        width=1720,
                                        height=120):
        """
        创建双语字幕
        """

        try:
            import tempfile
            from PIL import Image, ImageDraw, ImageFont

            zh_font_size = 32
            en_font_size = 22
            zh_en_gap = 6

            # 生成中文字幕
            zh_img, zh_height = create_subtitle_image(zh_text, width, height,
                                                      zh_font_size, 'black')

            # 生成英文字幕
            if en_text.strip():
                en_img, en_height = create_subtitle_image(en_text, width, height,
                                                          en_font_size, 'gray')
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

            temp_path = os.path.join(tempfile.gettempdir(),
                                     f'subtitle_{uuid.uuid4()}.png')
            final_img.save(temp_path)
            print(f'[字幕生成] 双语字幕图片已保存到: {temp_path}')
            return temp_path, final_height
        except Exception as e:
            print(f'字幕生成失败: {e}')
            try:
                import tempfile
                from PIL import Image, ImageDraw, ImageFont
                img = Image.new('RGBA', (width, 100), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                font = ImageFont.load_default()
                draw.text((50, 30), zh_text[:50], fill=(255, 255, 255), font=font)
                temp_path = os.path.join(tempfile.gettempdir(),
                                         f'subtitle_fallback_{uuid.uuid4()}.png')
                img.save(temp_path)
                print(f'[字幕生成] 回退字幕图片已保存到: {temp_path}')
                return temp_path, 100
            except:  # noqa
                return '', 100

    def _generate_assets_from_script(self, segments) -> str:
        # Generate bilingual subtitles
        def _split_subtitles(text: str, max_chars: int = 30) -> List[str]:
            import re
            sentences = re.split(r'([。！？；，、])', text)
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

        for i, seg in enumerate(segments):
            try:
                if seg.get('type') != 'text':
                    zh_text = seg.get('explanation', '') or seg.get(
                        'content', '')
                    parts = _split_subtitles(zh_text, max_chars=30)
                    img_list = []
                    for idx_p, part in enumerate(parts):
                        sub_en = self.translate_text_to_english(part)
                        temp_path, _h = self.create_bilingual_subtitle_image(
                            zh_text=part,
                            en_text=sub_en,
                            width=1720,
                            height=120)
                        if temp_path and os.path.exists(temp_path):
                            final_sub_path = os.path.join(
                                subtitle_dir,
                                f'bilingual_subtitle_{i+1}_{idx_p+1}.png')
                            try:
                                os.replace(temp_path, final_sub_path)
                            except Exception:
                                import shutil
                                shutil.move(temp_path, final_sub_path)
                            img_list.append(final_sub_path)
                    asset_paths['subtitle_segments_list'][i] = img_list
                    asset_paths['subtitle_paths'][
                        i] = img_list[0] if img_list else None
                else:
                    zh_text = seg.get('content', '')
                    en_text = self.translate_text_to_english(zh_text)
                    temp_path, _h = self.create_bilingual_subtitle_image(
                        zh_text=zh_text,
                        en_text=en_text,
                        width=1720,
                        height=120)
                    if temp_path and os.path.exists(temp_path):
                        final_sub_path = os.path.join(
                            subtitle_dir, f'bilingual_subtitle_{i+1}.png')
                        try:
                            os.replace(temp_path, final_sub_path)
                        except Exception:
                            import shutil
                            shutil.move(temp_path, final_sub_path)
                        asset_paths['subtitle_paths'][i] = final_sub_path
                        asset_paths['subtitle_segments_list'][i] = [
                            final_sub_path
                        ]
            except Exception as e:
                print(
                    f'[video_agent] Subtitle generation failed at segment {i+1}: {e}'
                )

        # Save all necessary info for the next step
        asset_info = {
            'topic': topic,
            'output_dir': full_output_dir,
            'segments': segments,
            'asset_paths': asset_paths,
            'animation_mode': self.animation_mode
        }
        asset_info_path = os.path.join(full_output_dir, 'asset_info.json')
        with open(asset_info_path, 'w', encoding='utf-8') as f:
            json.dump(asset_info, f, ensure_ascii=False, indent=2)

        # 兼容工作室的完整合成：同时输出 segments.json
        try:
            with open(
                    os.path.join(full_output_dir, 'segments.json'),
                    'w',
                    encoding='utf-8') as sf:
                json.dump(segments, sf, ensure_ascii=False, indent=2)
        except Exception as _e:
            print(f'[video_agent] 写入 segments.json 失败: {_e}')

        # In human mode, drop a short README to guide manual studio
        if self.animation_mode == 'human':
            readme_path = os.path.join(full_output_dir, 'HUMAN_README.txt')
            try:
                with open(readme_path, 'w', encoding='utf-8') as rf:
                    rf.write(
                        '本目录为人工动画模式生成的素材预备目录\n'
                        '- 已生成脚本、语音、插画、字幕与占位前景（无自动动画）\n'
                        '- 下一步：进入互动动画工作室制作每个动画片段\n\n'
                        '启动命令示例：\n'
                        '# 先确保将 ms-agent 目录加入 PYTHONPATH 环境变量\n'
                        '# PowerShell:\n'
                        "# $env:PYTHONPATH=\"{}\"\n"
                        '# 然后以模块方式启动工作室：\n'
                        "python -m projects.video_generate.core.human_animation_studio \"{}\"\n"
                        .format(
                            os.path.abspath(
                                os.path.join(
                                    os.path.dirname(__file__), '..',
                                    '..')),  # ms-agent 根目录
                            full_output_dir))
            except Exception as _e:
                print(f'[video_agent] Failed to write HUMAN_README: {_e}')

        print(
            f'[video_agent] Asset generation complete. Info saved to {asset_info_path}'
        )
        return asset_info_path

    async def run(self, inputs: Union[str, List[Message]],
                  **kwargs) -> List[Message]:

