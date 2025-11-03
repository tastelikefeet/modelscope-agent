import re
from dataclasses import dataclass, field
import json
import os
from typing import List, Union

from omegaconf import DictConfig

from ms_agent.agent import CodeAgent
from ms_agent.llm import Message
from projects.video_generate.core import workflow as video_workflow
from ms_agent.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Pattern:

    name: str
    pattern: str
    tags: List[str] = field(default_factory=list)


class GenerateAssets(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.animation_mode = os.environ.get('MS_ANIMATION_MODE',
                                              'auto').strip().lower() or 'auto'
        self.patterns = self.create_patterns()

    @staticmethod
    def create_patterns():
        patterns = [Pattern(name='formula', pattern=r'<formula>(.*?)</formula>', tags=['<formula>', '</formula>']),
                    Pattern(name='code', pattern=r'<code>(.*?)</code>', tags=['<code>', '</code>']),
                    Pattern(name='chart', pattern=r'<chart>(.*?)</chart>', tags=['<chart>', '</chart>']),
                    Pattern(name='definition', pattern=r'<definition>(.*?)</definition>',
                            tags=['<definition>', '</definition>']),
                    Pattern(name='theorem', pattern=r'<theorem>(.*?)</theorem>', tags=['<theorem>', '</theorem>']),
                    Pattern(name='example', pattern=r'<example>(.*?)</example>', tags=['<example>', '</example>']),
                    Pattern(name='emphasis', pattern=r'<emphasis>(.*?)</emphasis>', tags=['<emphasis>', '</emphasis>']),
                    Pattern(name='comparison', pattern=r'<comparison>(.*?)</comparison>',
                            tags=['<comparison>', '</comparison>']),
                    Pattern(name='step', pattern=r'<step>(.*?)</step>', tags=['<step>', '</step>']),
                    Pattern(name='metaphor', pattern=r'<metaphor>([^<]*?)(?=<|$)', tags=['<metaphor>']),
                    Pattern(name='analogy', pattern=r'<analogy>([^<]*?)(?=<|$)', tags=['<analogy>']),
                    Pattern(name='note', pattern=r'<note>([^<]*?)(?=<|$)', tags=['<note>']),
                    Pattern(name='tip', pattern=r'<tip>([^<]*?)(?=<|$)', tags=['<tip>']),
                    Pattern(name='key', pattern=r'<key>([^<]*?)(?=<|$)', tags=['<key>'])]
        return patterns

    def analyze_content_context(content, content_type, surrounding_text=''):
        context_info = {
            'emphasis_words': [],
            'explanation_flow': [],
            'timing_cues': [],
            'emotional_tone': 'neutral',
            'complexity_level': 'medium'
        }

        emphasis_patterns = [
            r'【强调】(.*?)【/强调】', r'重要的是', r'关键在于', r'特别注意', r'核心概念', r'最重要的'
        ]

        for pattern in emphasis_patterns:
            matches = re.findall(pattern, surrounding_text, re.IGNORECASE)
            context_info['emphasis_words'].extend(matches)

        flow_patterns = [r'首先|第一', r'然后|接着|其次', r'最后|最终|总之', r'例如|比如', r'因此|所以']

        for pattern in flow_patterns:
            if re.search(pattern, surrounding_text):
                context_info['explanation_flow'].append(pattern)

        if len(content
               ) > 100 or '复杂' in surrounding_text or '高级' in surrounding_text:
            context_info['complexity_level'] = 'high'
        elif len(content
                 ) < 30 or '简单' in surrounding_text or '基础' in surrounding_text:
            context_info['complexity_level'] = 'low'

        if any(word in surrounding_text for word in ['激动', '兴奋', '惊人', '突破']):
            context_info['emotional_tone'] = 'excited'
        elif any(word in surrounding_text for word in ['重要', '关键', '核心', '必须']):
            context_info['emotional_tone'] = 'serious'
        elif any(word in surrounding_text for word in ['简单', '容易', '轻松']):
            context_info['emotional_tone'] = 'casual'

        return context_info

    async def run(self, inputs: Union[str, List[Message]],
                  **kwargs) -> List[Message]:
        return await self._generate_assets_from_script(inputs)

    def parse_structured_content(self, script):
        segments = []
        current_pos = 0

        all_matches = []
        for item in self.patterns:
            for match in re.finditer(item.pattern, script, re.DOTALL):
                all_matches.append({
                    'start': match.start(),
                    'end': match.end(),
                    'type': item.name,
                    'content': match.group(1).strip(),
                    'full_match': match.group(0)
                })

        all_matches.sort(key=lambda x: x['start'])

        for i, match in enumerate(all_matches):
            if match['start'] > current_pos:
                normal_text = script[current_pos:match['start']].strip()
                if normal_text:
                    segments.append({'type': 'text', 'content': normal_text})

            context_start = max(0, match['start'] - 100)
            context_end = min(len(script), match['end'] + 100)
            surrounding_text = script[context_start:context_end]

            context_info = analyze_content_context(match['content'], match['type'],
                                                   surrounding_text)

            segments.append({
                'type': match['type'],
                'content': match['content'],
                'surrounding_text': surrounding_text,
                'context_info': context_info,
                'position_in_script': match['start'] / len(script)
            })

            current_pos = match['end']

        if current_pos < len(script):
            remaining_text = script[current_pos:].strip()
            if remaining_text:
                segments.append({'type': 'text', 'content': remaining_text})

        return segments

    async def _generate_assets_from_script(self, messages: List[Message]) -> str:
        logger.info('Starting asset generation from script')
        with open('script.txt', 'r', encoding='utf-8') as f:
            script = f.read()

        for message in messages:
            if message.role == 'user':
                topic = messages[1]
                break

        segments = video_workflow.parse_structured_content(script)

        # Further split long text segments
        final_segments = []
        for segment in segments:
            if segment['type'] == 'text' and len(segment['content']) > 100:
                subsegments = video_workflow.split_text_by_punctuation(
                    segment['content'])
                for subseg_dict in subsegments:
                    if subseg_dict['content'].strip():
                        final_segments.append({
                            'content':
                            subseg_dict['content'].strip(),
                            'type':
                            'text',
                            'parent_segment':
                            segment
                        })
            else:
                final_segments.append(segment)
        segments = final_segments
        print(f'[video_agent] Script parsed into {len(segments)} segments.')

        # 2. Generate assets for each segment
        asset_paths = {
            'audio_paths': [],
            'foreground_paths': [],
            'subtitle_paths': [],
            'illustration_paths': [],
            'subtitle_segments_list': []
        }

        tts_dir = os.path.join(full_output_dir, 'audio')
        os.makedirs(tts_dir, exist_ok=True)

        subtitle_dir = os.path.join(full_output_dir, 'subtitles')
        os.makedirs(subtitle_dir, exist_ok=True)

        # Prepare illustration paths list aligned to segments
        illustration_paths: List[str] = []

        for i, segment in enumerate(segments):
            print(
                f"[video_agent] Processing segment {i+1}/{len(segments)}: {segment['type']}"
            )

            # Clean content to avoid issues with markers
            tts_text = video_workflow.clean_content(segment.get('content', ''))

            # Generate TTS
            audio_path = os.path.join(tts_dir, f'segment_{i+1}.mp3')
            if tts_text:
                if video_workflow.edge_tts_generate(tts_text, audio_path):
                    segment[
                        'audio_duration'] = video_workflow.get_audio_duration(
                            audio_path)
                else:
                    video_workflow.create_silent_audio(
                        audio_path, duration=3.0)
                    segment['audio_duration'] = 3.0
            else:
                video_workflow.create_silent_audio(audio_path, duration=2.0)
                segment['audio_duration'] = 2.0
            asset_paths['audio_paths'].append(audio_path)

            # Generate Animation (only for non-text types)
            if segment['type'] != 'text' and self.animation_mode != 'human':
                manim_code = video_workflow.generate_manim_code(
                    content=video_workflow.clean_content(segment['content']),
                    content_type=segment['type'],
                    scene_number=i + 1,
                    audio_duration=segment.get('audio_duration', 8.0),
                    main_theme=topic,
                    context_segments=segments,
                    segment_index=i,
                    total_segments=segments)
                video_path = None
                if manim_code:
                    scene_name = f'Scene{i+1}'
                    scene_dir = os.path.join(full_output_dir, f'scene_{i+1}')
                    video_path = video_workflow.render_manim_scene(
                        manim_code, scene_name, scene_dir)
                asset_paths['foreground_paths'].append(video_path)
            else:
                # In human mode, skip auto manim rendering (leave placeholders)
                asset_paths['foreground_paths'].append(None)

            # Initialize placeholders for subtitles; will fill after loop
            illustration_paths.append(None)
            asset_paths['subtitle_paths'].append(None)
            asset_paths['subtitle_segments_list'].append([])

        # Generate illustrations for text segments (mirrors original logic)
        try:
            text_segments = [
                seg for seg in segments if seg.get('type') == 'text'
            ]
            if text_segments:
                illustration_prompts_path = os.path.join(
                    full_output_dir, 'illustration_prompts.json')
                if os.path.exists(illustration_prompts_path):
                    illustration_prompts = json.load(
                        open(illustration_prompts_path, 'r', encoding='utf-8'))
                else:
                    illustration_prompts = video_workflow.generate_illustration_prompts(
                        [seg['content'] for seg in text_segments])
                    json.dump(
                        illustration_prompts,
                        open(illustration_prompts_path, 'w', encoding='utf-8'),
                        ensure_ascii=False,
                        indent=2)

                images_dir = os.path.join(full_output_dir, 'images')
                os.makedirs(images_dir, exist_ok=True)
                image_paths_path = os.path.join(images_dir, 'image_paths.json')
                if os.path.exists(image_paths_path):
                    image_paths = json.load(
                        open(image_paths_path, 'r', encoding='utf-8'))
                else:
                    image_paths = video_workflow.generate_images(
                        illustration_prompts, output_dir=full_output_dir)
                    # move to images folder for consistent paths
                    for i, img_path in enumerate(image_paths):
                        if os.path.exists(img_path):
                            new_path = os.path.join(
                                images_dir, f'illustration_{i+1}.png'
                                if img_path.lower().endswith('.png') else
                                f'illustration_{i+1}.jpg')
                            try:
                                os.replace(img_path, new_path)
                            except Exception:
                                try:
                                    import shutil
                                    shutil.move(img_path, new_path)
                                except Exception:
                                    new_path = img_path
                            image_paths[i] = new_path
                    json.dump(
                        image_paths,
                        open(image_paths_path, 'w', encoding='utf-8'),
                        ensure_ascii=False,
                        indent=2)

                fg_out_dir = os.path.join(images_dir, 'output_black_only')
                os.makedirs(fg_out_dir, exist_ok=True)
                # process background removal if needed
                if len([
                        f for f in os.listdir(fg_out_dir)
                        if f.lower().endswith('.png')
                ]) < len(image_paths):
                    video_workflow.keep_only_black_for_folder(
                        images_dir, fg_out_dir)

                # map illustrations back to segment indices
                text_idx = 0
                for idx, seg in enumerate(segments):
                    if seg.get('type') == 'text':
                        if text_idx < len(image_paths):
                            transparent_path = os.path.join(
                                fg_out_dir, f'illustration_{text_idx+1}.png')
                            if os.path.exists(transparent_path):
                                illustration_paths[idx] = transparent_path
                            else:
                                illustration_paths[idx] = image_paths[text_idx]
                            text_idx += 1
                        else:
                            illustration_paths[idx] = None
                    else:
                        illustration_paths[idx] = None
            else:
                illustration_paths = [None] * len(segments)
        except Exception as e:
            print(f'[video_agent] Illustration generation failed: {e}')
            illustration_paths = [None] * len(segments)

        # Attach illustration paths to asset_paths
        asset_paths['illustration_paths'] = illustration_paths

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
                        sub_en = video_workflow.translate_text_to_english(part)
                        temp_path, _h = video_workflow.create_bilingual_subtitle_image(
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
                    en_text = video_workflow.translate_text_to_english(zh_text)
                    temp_path, _h = video_workflow.create_bilingual_subtitle_image(
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
