import os
from typing import List, Union

import json
from ms_agent.agent.base import Agent
from ms_agent.llm import Message
from omegaconf import DictConfig
from projects.video_generate.core import workflow as video_workflow


class ComposeVideo(Agent):
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

    def compose_final_video(background_path, foreground_paths, audio_paths,
                            subtitle_paths, illustration_paths, segments,
                            output_path, subtitle_segments_list):
        """
        合成插画+动画的最终视频
        """

        try:
            import moviepy.editor as mp

            print('开始合成最终视频...')
            segment_durations = []
            total_duration = 0

            for i, audio_path in enumerate(audio_paths):
                actual_duration = 3.0

                if audio_path and os.path.exists(audio_path):
                    try:
                        audio_clip = mp.AudioFileClip(audio_path)
                        actual_duration = max(audio_clip.duration, 3.0)
                        audio_clip.close()
                    except:  # noqa
                        actual_duration = 3.0

                if i < len(foreground_paths
                           ) and foreground_paths[i] and os.path.exists(
                    foreground_paths[i]):
                    try:
                        animation_clip = mp.VideoFileClip(
                            foreground_paths[i], has_mask=True)
                        animation_duration = animation_clip.duration
                        animation_clip.close()

                        if animation_duration > actual_duration:
                            actual_duration = animation_duration
                            print(f'片段 {i + 1} 使用动画时长: {actual_duration:.1f}秒')
                    except:  # noqa
                        pass

                segment_durations.append(actual_duration)
                total_duration += actual_duration

            print(f'总时长: {total_duration:.1f}秒，{len(segment_durations)}个片段')
            print('重新组织合成逻辑...')

            print('步骤1：合成每个片段的完整视频...')
            segment_videos = []

            for i, (duration,
                    segment) in enumerate(zip(segment_durations, segments)):
                print(
                    f"合成片段 {i + 1}: {segment.get('type', 'unknown')} - {duration:.1f}秒"
                )

                current_video_clips = []

                if background_path and os.path.exists(background_path):
                    bg_clip = mp.ImageClip(background_path, duration=duration)
                    bg_clip = bg_clip.resize((1920, 1080))
                    current_video_clips.append(bg_clip)

                if segment.get('type') == 'text' and i < len(
                        illustration_paths
                ) and illustration_paths[i] and os.path.exists(
                    illustration_paths[i]):
                    try:
                        illustration_clip = mp.ImageClip(
                            illustration_paths[i], duration=duration)
                        original_w, original_h = illustration_clip.size
                        available_w, available_h = 1920, 800
                        scale_w = available_w / original_w
                        scale_h = available_h / original_h
                        scale = min(scale_w, scale_h, 1.0)

                        if scale < 1.0:
                            new_w = int(original_w * scale)
                            new_h = int(original_h * scale)
                            illustration_clip = illustration_clip.resize(
                                (new_w, new_h))
                        else:
                            new_w, new_h = original_w, original_h

                        # 向左运动动画
                        exit_duration = 1.0
                        start_animation_time = max(duration - exit_duration, 0)
                        print(
                            f'调试: 片段时长={duration:.2f}秒, 退出动画时长={exit_duration}秒, 动画开始时间={start_animation_time:.2f}秒'
                        )
                        print(
                            f'调试: 插画静止时间={start_animation_time:.2f}秒, 动画运动时间={exit_duration}秒'
                        )

                        def illustration_pos_factory(idx, start_x, end_x, new_h,
                                                     start_animation_time,
                                                     exit_duration):

                            def illustration_pos(t):
                                y = (1080 - new_h) // 2
                                if t < start_animation_time:
                                    x = start_x
                                    print(
                                        f'调试: 片段{idx}  时间={t:.2f}秒, 静止位置=({x}, {y})，动画将在{start_animation_time:.2f}秒开始'
                                    )
                                elif t < start_animation_time + exit_duration:
                                    progress = (
                                                       t - start_animation_time) / exit_duration
                                    progress = min(max(progress, 0), 1)  # 限制在0~1
                                    x = start_x + (end_x - start_x) * progress
                                    print(
                                        f'调试: 片段{idx}  时间={t:.2f}秒, 运动位置=({x}, {y})，进度={progress:.1%}'
                                    )
                                else:
                                    x = end_x
                                    print(
                                        f'调试: 片段{idx}  时间={t:.2f}秒, 已运动结束，插画在屏幕外 ({x}, {y})'
                                    )
                                return (x, y)

                            return illustration_pos

                        print(
                            f'插画动画设置: 片段时长 {duration:.1f}秒，动画在最后 {exit_duration}秒开始'
                        )
                        illustration_clip = illustration_clip.set_position(
                            illustration_pos_factory(i, (1920 - new_w) // 2,
                                                     -new_w, new_h,
                                                     start_animation_time,
                                                     exit_duration))
                        current_video_clips.append(illustration_clip)
                        print('添加插画层')
                    except Exception as e:
                        print(f'插画加载失败: {e}')

                elif segment.get('type') != 'text' and i < len(
                        foreground_paths
                ) and foreground_paths[i] and os.path.exists(foreground_paths[i]):
                    try:
                        fg_clip = mp.VideoFileClip(
                            foreground_paths[i], has_mask=True)
                        original_w, original_h = fg_clip.size
                        available_w, available_h = 1920, 800
                        scale_w = available_w / original_w
                        scale_h = available_h / original_h
                        scale = min(scale_w, scale_h, 1.0)

                        if scale < 1.0:
                            new_w = int(original_w * scale)
                            new_h = int(original_h * scale)
                            fg_clip = fg_clip.resize((new_w, new_h))

                        fg_clip = fg_clip.set_position(('center', 'center'))
                        fg_clip = fg_clip.set_duration(duration)
                        current_video_clips.append(fg_clip)
                        print('添加动画层')
                    except Exception as e:
                        print(f'动画加载失败: {e}')

                if segment.get('type') != 'text' and i < len(
                        subtitle_segments_list):
                    try:
                        subtitle_imgs = subtitle_segments_list[i]
                        if subtitle_imgs and isinstance(
                                subtitle_imgs, list) and len(subtitle_imgs) > 0:
                            n = len(subtitle_imgs)
                            seg_duration = duration / n
                            for idx, subtitle_path in enumerate(subtitle_imgs):
                                try:
                                    from PIL import Image
                                    subtitle_img = Image.open(subtitle_path)
                                    subtitle_w, subtitle_h = subtitle_img.size
                                    subtitle_clip = mp.ImageClip(
                                        subtitle_path, duration=seg_duration)
                                    subtitle_clip = subtitle_clip.resize(
                                        (subtitle_w, subtitle_h))
                                    subtitle_y = 850
                                    print(f'字幕位置设置为 y={subtitle_y}')
                                    subtitle_clip = subtitle_clip.set_position(
                                        ('center', subtitle_y))
                                    subtitle_clip = subtitle_clip.set_start(
                                        idx * seg_duration)
                                    current_video_clips.append(subtitle_clip)
                                    print(f'添加动画片段字幕 {idx + 1}/{n}')
                                except Exception as e:
                                    print(f'动画片段字幕 {idx + 1} 处理失败: {e}')
                        else:
                            print(f'动画片段 {i + 1} 没有有效字幕，跳过字幕层')
                    except Exception as e:
                        print(f'动画片段 {i + 1} 字幕处理异常: {e}')
                else:
                    if i < len(subtitle_paths
                               ) and subtitle_paths[i] and os.path.exists(
                        subtitle_paths[i]):
                        try:
                            from PIL import Image
                            subtitle_img = Image.open(subtitle_paths[i])
                            subtitle_w, subtitle_h = subtitle_img.size
                            subtitle_clip = mp.ImageClip(
                                subtitle_paths[i], duration=duration)
                            subtitle_clip = subtitle_clip.resize(
                                (subtitle_w, subtitle_h))
                            subtitle_y = 850
                            print(f'字幕位置设置为 y={subtitle_y}')
                            subtitle_clip = subtitle_clip.set_position(
                                ('center', subtitle_y))
                            current_video_clips.append(subtitle_clip)
                            print('添加字幕层（底部对齐）')
                        except Exception as e:
                            print(f'字幕加载失败: {e}')

                if current_video_clips:
                    segment_video = mp.CompositeVideoClip(
                        current_video_clips, size=(1920, 1080))
                    segment_videos.append(segment_video)
                    print(f'片段 {i + 1} 合成完成')
                else:
                    print(f'片段 {i + 1} 无有效内容，跳过')

            if not segment_videos:
                print('没有有效的视频片段')
                return None

            print('  步骤2：按时间顺序连接所有片段...')
            final_video = mp.concatenate_videoclips(
                segment_videos, method='compose')
            print(f'视频连接完成，总时长: {final_video.duration:.1f}秒')

            print('步骤3：合成音频...')
            if audio_paths:
                try:
                    print(f'连接 {len(audio_paths)} 个音频片段...')
                    valid_audio_clips = []
                    for i, (audio_path, duration) in enumerate(
                            zip(audio_paths, segment_durations)):
                        try:
                            if audio_path and os.path.exists(audio_path):
                                audio_clip = mp.AudioFileClip(audio_path)
                                audio_clip = audio_clip.set_fps(44100)
                                try:
                                    audio_clip = audio_clip.set_channels(2)
                                except Exception:
                                    pass
                                if audio_clip.duration > duration:
                                    audio_clip = audio_clip.subclip(0, duration)
                                elif audio_clip.duration < duration:
                                    from moviepy.editor import AudioClip
                                    silence = AudioClip(
                                        lambda t: [0, 0],
                                        duration=duration
                                                 - audio_clip.duration).set_fps(44100)
                                    try:
                                        silence = silence.set_channels(2)
                                    except Exception:
                                        pass
                                    audio_clip = mp.concatenate_audioclips(
                                        [audio_clip, silence])
                                valid_audio_clips.append(audio_clip)
                                print(f'音频片段 {i + 1}: {audio_clip.duration:.2f}s')
                            else:
                                print(f'音频片段 {i + 1} 无效，跳过')
                        except Exception as e:
                            print(f'音频片段 {i + 1} 处理失败: {e}')
                            from moviepy.editor import AudioClip
                            silence = AudioClip(
                                lambda t: [0, 0], duration=duration).set_fps(44100)
                            valid_audio_clips.append(silence)

                    if valid_audio_clips:
                        final_audio = mp.concatenate_audioclips(valid_audio_clips)
                        print(f'音频连接完成，总时长: {final_audio.duration:.1f}秒')
                        if final_audio.duration > final_video.duration:
                            final_audio = final_audio.subclip(
                                0, final_video.duration)
                            print('音频已裁剪到视频时长')
                        elif final_audio.duration < final_video.duration:
                            from moviepy.editor import AudioClip
                            silence = AudioClip(
                                lambda t: [0, 0],
                                duration=final_video.duration
                                         - final_audio.duration)
                            final_audio = mp.concatenate_audioclips(
                                [final_audio, silence])
                            print('音频已补足到视频时长')

                        final_video = final_video.set_audio(final_audio)
                        print(
                            f'音频合成成功，时长: {final_audio.duration:.1f}秒 (视频: {final_video.duration:.1f}秒)'
                        )
                    else:
                        print('没有有效音频，生成静音视频')
                except Exception as e:
                    print(f'音频合成失败: {e}')
                    print('将生成无声视频')
            else:
                print('没有音频片段，生成静音视频')

            try:
                import moviepy.audio.fx.all as afx
                bg_music_path = os.path.join(
                    os.path.dirname(__file__), 'asset', 'bg_audio.mp3')
                if os.path.exists(bg_music_path):
                    print('添加背景音乐...')
                    bg_music = mp.AudioFileClip(bg_music_path)
                    bg_music = afx.audio_loop(
                        bg_music, duration=final_video.duration)
                    bg_music = bg_music.volumex(0.2)
                    if final_video.audio:
                        tts_audio = final_video.audio.set_duration(
                            final_video.duration).volumex(1.0)
                        bg_audio = bg_music.set_duration(
                            final_video.duration).volumex(0.15)
                        mixed_audio = mp.CompositeAudioClip(
                            [tts_audio,
                             bg_audio]).set_duration(final_video.duration)
                    else:
                        mixed_audio = bg_music.set_duration(
                            final_video.duration).volumex(0.3)
                    final_video = final_video.set_audio(mixed_audio)
                    print('背景音乐添加完成')
                else:
                    print('未找到背景音乐文件')
            except Exception as e:
                print(f'背景音乐添加失败: {e}')

            print('渲染最终视频...')
            if final_video is None:
                print('错误: final_video为None，无法渲染')
                return None

            try:
                print(f'视频总时长: {final_video.duration:.1f}秒')
                print(f'视频分辨率: {final_video.size}')
                print(f"音频状态: {'有音频' if final_video.audio else '无音频'}")
                print(f'final_video类型: {type(final_video)}')
                print(f'final_video属性: {dir(final_video)}')

                if final_video.audio:
                    print(f'音频类型: {type(final_video.audio)}')
                    print(f'音频时长: {final_video.audio.duration:.1f}秒')
                    try:
                        audio_fps = final_video.audio.fps
                        print(f'音频采样率: {audio_fps} Hz')
                    except AttributeError:
                        if hasattr(final_video.audio,
                                   'clips') and final_video.audio.clips:
                            first_clip = final_video.audio.clips[0]
                            if hasattr(first_clip, 'fps'):
                                print(f'首个音频片段采样率: {first_clip.fps} Hz')
            except Exception as e:
                print(f'错误: 获取视频信息失败: {e}')
                import traceback
                traceback.print_exc()
                return None

            try:
                print(f'开始渲染到: {output_path}')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                final_video.write_videofile(
                    output_path,
                    fps=24,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    logger=None,
                    verbose=False,
                    threads=2,
                    bitrate='5000k',
                    audio_bitrate='192k',
                    audio_fps=44100,
                    write_logfile=False)

                print(f'视频渲染完成: {output_path}')
                if os.path.exists(
                        output_path) and os.path.getsize(output_path) > 1024:
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f'文件大小: {file_size_mb:.1f} MB')

                    try:
                        test_clip = mp.VideoFileClip(output_path)
                        actual_duration = test_clip.duration
                        test_clip.close()
                        print(
                            f'实际时长: {actual_duration:.1f}秒 (预期: {final_video.duration:.1f}秒)'
                        )

                        if abs(actual_duration - final_video.duration) < 1.0:
                            print('视频文件验证通过')
                            return output_path
                        else:
                            print('视频时长不匹配，但文件已生成')
                            return output_path
                    except Exception as e:
                        print(f'视频文件验证失败: {e}')
                        return output_path
                else:
                    print('视频文件生成失败或文件过小')
                    return None

            except Exception as e:
                print(f'视频渲染失败: {e}')
                import traceback
                traceback.print_exc()

                try:
                    print('尝试生成无音频视频...')
                    final_video = final_video.set_audio(None)
                    final_video.write_videofile(
                        output_path,
                        fps=24,
                        codec='libx264',
                        audio_codec=None,
                        temp_audiofile='temp-audio.m4a',
                        remove_temp=True,
                        logger=None,
                        verbose=False,
                        threads=2,
                        bitrate='5000k',
                        write_logfile=False)
                    print(f'无音频视频渲染完成: {output_path}')
                    return output_path
                except Exception as e2:
                    print(f'无音频视频渲染也失败: {e2}')
                    traceback.print_exc()
                    return None

        except Exception as e:
            print(f'视频合成失败: {e}')
            return None

    async def run(self, inputs: Union[str, List[Message]],
                  **kwargs) -> List[Message]:
        # Compose final video (human mode generates a preview with placeholders)
        final_name = 'preview_with_placeholders.mp4' if self.animation_mode == 'human' else 'final_video.mp4'
        final_video_path = os.path.join(full_output_dir, final_name)

        composed_path = self.compose_final_video(
            background_path=background_path,
            foreground_paths=asset_paths['foreground_paths'],
            audio_paths=asset_paths['audio_paths'],
            subtitle_paths=asset_paths['subtitle_paths'],
            illustration_paths=asset_paths['illustration_paths'],
            segments=segments,
            output_path=final_video_path,
            subtitle_segments_list=asset_paths['subtitle_segments_list'])

        if composed_path and os.path.exists(composed_path):
            print(
                f'[video_agent] Final video successfully composed at: {composed_path}'
            )
            return composed_path
        else:
            print('[video_agent] Final video composition failed.')
            return None
