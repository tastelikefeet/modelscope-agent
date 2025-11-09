import os

import moviepy as afx
import moviepy as mp
from moviepy import AudioClip, vfx

from ms_agent.agent import CodeAgent
from ms_agent.utils import get_logger
from omegaconf import DictConfig, ListConfig
from PIL import Image

logger = get_logger()


class ComposeVideo(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.animation_mode = getattr(self.config, 'animation_code', 'auto')
        self.transition = getattr(self.config, 't2i_transition', 'ken-burns-effect')

    def compose_final_video(self, background_path, foreground_paths,
                            audio_paths, subtitle_paths, illustration_paths,
                            segments, output_path, subtitle_segments_list):
        segment_durations = []
        total_duration = 0
        logger.info(f'Composing the final video.')

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
                animation_clip = mp.VideoFileClip(
                    foreground_paths[i], has_mask=True)
                animation_duration = animation_clip.duration
                animation_clip.close()

                if animation_duration > actual_duration:
                    actual_duration = animation_duration

            segment_durations.append(actual_duration)
            total_duration += actual_duration

        logger.info(
            f'Total duration: {total_duration:.1f}s，{len(segment_durations)} segments.'
        )
        logger.info('Step1: Compose video for each segment.')
        segment_videos = []

        for i, (duration,
                segment) in enumerate(zip(segment_durations, segments)):
            logger.info(
                f"Processing {i + 1} segment: {segment.get('type', 'unknown')} - {duration:.1f} seconds."
            )

            current_video_clips = []

            if background_path and os.path.exists(background_path):
                bg_clip = mp.ImageClip(background_path, duration=duration)
                bg_clip = bg_clip.resized((1920, 1080))
                current_video_clips.append(bg_clip)

            if segment.get('type') == 'text' and i < len(
                    illustration_paths
            ) and illustration_paths[i] and os.path.exists(
                    illustration_paths[i]):

                illustration_clip = mp.ImageClip(
                    illustration_paths[i], duration=duration)
                original_w, original_h = illustration_clip.size
                available_w, available_h = 1920, 1080
                scale_w = available_w / original_w
                scale_h = available_h / original_h
                scale = min(scale_w, scale_h, 1.0)

                if scale < 1.0:
                    new_w = int(original_w * scale)
                    new_h = int(original_h * scale)
                    illustration_clip = illustration_clip.resized(
                        (new_w, new_h))
                else:
                    new_w, new_h = original_w, original_h

                exit_duration = 1.0
                start_animation_time = max(duration - exit_duration, 0)

                if self.transition == 'ken-burns-effect':
                    # Ken Burns effect: smooth and stable zoom-in
                    zoom_factor = 1.08  # Gentle zoom from 1.0 to 1.08x (reduced from 1.2x)
                    
                    def ken_burns_factory(idx, new_w, new_h, duration, zoom_factor):
                        def resize_effect(get_frame, t):
                            # Calculate zoom progress with easing (0 to 1)
                            progress = t / duration
                            # Ease-in-out function for smoother animation
                            eased_progress = progress * progress * (3.0 - 2.0 * progress)
                            current_zoom = 1.0 + (zoom_factor - 1.0) * eased_progress
                            
                            # Get original frame
                            frame = get_frame(t)
                            from PIL import Image
                            import numpy as np
                            
                            # Convert to PIL for easier manipulation
                            img = Image.fromarray(frame)
                            orig_w, orig_h = img.size
                            
                            # Calculate zoomed size
                            zoomed_w = int(orig_w * current_zoom)
                            zoomed_h = int(orig_h * current_zoom)
                            
                            # Resize image - use LANCZOS for better quality
                            try:
                                # Try newer Pillow API first
                                img_zoomed = img.resize((zoomed_w, zoomed_h), Image.Resampling.LANCZOS)
                            except AttributeError:
                                # Fallback to older Pillow API (numeric constant)
                                img_zoomed = img.resize((zoomed_w, zoomed_h), 1)  # 1 = LANCZOS
                            
                            # Keep image centered - no horizontal pan to avoid shake
                            max_offset_x = zoomed_w - orig_w
                            max_offset_y = zoomed_h - orig_h
                            
                            # Center the crop area (no panning movement)
                            offset_x = max_offset_x // 2
                            offset_y = max_offset_y // 2
                            
                            # Crop to original size
                            img_cropped = img_zoomed.crop((
                                offset_x,
                                offset_y,
                                offset_x + orig_w,
                                offset_y + orig_h
                            ))
                            
                            return np.array(img_cropped)
                        
                        return resize_effect
                    
                    # Apply Ken Burns effect
                    illustration_clip = illustration_clip.transform(
                        ken_burns_factory(i, new_w, new_h, duration, zoom_factor)
                    )
                    illustration_clip = illustration_clip.with_position('center')
                    
                    # Add fade in/out effects for smooth transitions
                    fade_duration = min(0.8, duration / 3)
                    illustration_clip = illustration_clip.with_effects([
                        vfx.CrossFadeIn(fade_duration),
                        vfx.CrossFadeOut(fade_duration)
                    ])
                    
                elif self.transition == 'slide':
                    # Default slide left animation
                    def illustration_pos_factory(idx, start_x, end_x, new_h,
                                                 start_animation_time,
                                                 exit_duration):

                        def illustration_pos(t):
                            y = (1080 - new_h) // 2
                            if t < start_animation_time:
                                x = start_x
                            elif t < start_animation_time + exit_duration:
                                progress = (t
                                            - start_animation_time) / exit_duration
                                progress = min(max(progress, 0), 1)
                                x = start_x + (end_x - start_x) * progress
                            else:
                                x = end_x
                            return x, y

                        return illustration_pos

                    illustration_clip = illustration_clip.with_position(
                        illustration_pos_factory(i, (1920 - new_w) // 2, -new_w,
                                                 new_h, start_animation_time,
                                                 exit_duration))
                
                current_video_clips.append(illustration_clip)

            elif segment.get('type') != 'text' and i < len(
                    foreground_paths
            ) and foreground_paths[i] and os.path.exists(foreground_paths[i]):
                fg_clip = mp.VideoFileClip(foreground_paths[i], has_mask=True)
                original_w, original_h = fg_clip.size
                available_w, available_h = 1920, 800
                scale_w = available_w / original_w
                scale_h = available_h / original_h
                scale = min(scale_w, scale_h, 1.0)

                if scale < 1.0:
                    new_w = int(original_w * scale)
                    new_h = int(original_h * scale)
                    fg_clip = fg_clip.resized((new_w, new_h))

                fg_clip = fg_clip.with_position(('center', 'center'))
                fg_clip = fg_clip.with_duration(duration)
                current_video_clips.append(fg_clip)

            if segment.get('type') != 'text' and i < len(
                    subtitle_segments_list):
                subtitle_imgs = subtitle_segments_list[i]
                if subtitle_imgs and isinstance(
                        subtitle_imgs, list) and len(subtitle_imgs) > 0:
                    n = len(subtitle_imgs)
                    seg_duration = duration / n
                    for idx, subtitle_path in enumerate(subtitle_imgs):
                        subtitle_img = Image.open(subtitle_path)
                        subtitle_w, subtitle_h = subtitle_img.size
                        subtitle_clip = mp.ImageClip(
                            subtitle_path, duration=seg_duration)
                        subtitle_clip = subtitle_clip.resized(
                            (subtitle_w, subtitle_h))
                        subtitle_y = 850
                        subtitle_clip = subtitle_clip.with_position(
                            ('center', subtitle_y))
                        subtitle_clip = subtitle_clip.set_start(idx
                                                                * seg_duration)
                        current_video_clips.append(subtitle_clip)
            else:
                if isinstance(subtitle_paths[i], (list, ListConfig)):
                    subtitle_paths[i] = subtitle_paths[i][0]
                if i < len(subtitle_paths
                           ) and subtitle_paths[i] and os.path.exists(
                               subtitle_paths[i]):
                    subtitle_img = Image.open(subtitle_paths[i])
                    subtitle_w, subtitle_h = subtitle_img.size
                    subtitle_clip = mp.ImageClip(
                        subtitle_paths[i], duration=duration)
                    subtitle_clip = subtitle_clip.resized(
                        (subtitle_w, subtitle_h))
                    subtitle_y = 850
                    subtitle_clip = subtitle_clip.with_position(
                        ('center', subtitle_y))
                    current_video_clips.append(subtitle_clip)

            if current_video_clips:
                segment_video = mp.CompositeVideoClip(
                    current_video_clips, size=(1920, 1080))
                segment_videos.append(segment_video)

        if not segment_videos:
            return None

        logger.info('Step2: Combine all video segments.')
        final_video = mp.concatenate_videoclips(
            segment_videos, method='compose')
        logger.info('Step3: Compose audios.')
        if audio_paths:
            valid_audio_clips = []
            for i, (audio_path, duration) in enumerate(
                    zip(audio_paths, segment_durations)):
                if audio_path and os.path.exists(audio_path):
                    audio_clip = mp.AudioFileClip(audio_path)
                    audio_clip = audio_clip.with_fps(44100)
                    # audio_clip = audio_clip.set_channels(2)
                    if audio_clip.duration > duration:
                        audio_clip = audio_clip.subclip(0, duration)
                    elif audio_clip.duration < duration:

                        silence = AudioClip(
                            lambda t: [0, 0],
                            duration=duration
                            - audio_clip.duration).with_fps(44100)
                        # silence = silence.set_channels(2)
                        audio_clip = mp.concatenate_audioclips(
                            [audio_clip, silence])
                    valid_audio_clips.append(audio_clip)

            if valid_audio_clips:
                final_audio = mp.concatenate_audioclips(valid_audio_clips)
                logger.info(
                    f'Audio composing done: {final_audio.duration:.1f} seconds.'
                )
                if final_audio.duration > final_video.duration:
                    final_audio = final_audio.subclip(0, final_video.duration)
                elif final_audio.duration < final_video.duration:
                    silence = AudioClip(
                        lambda t: [0, 0],
                        duration=final_video.duration - final_audio.duration)
                    final_audio = mp.concatenate_audioclips(
                        [final_audio, silence])

                final_video = final_video.with_audio(final_audio)

            bg_music_path = os.path.join(
                os.path.dirname(__file__), 'bg_audio.mp3')
            if os.path.exists(bg_music_path):
                bg_music = mp.AudioFileClip(bg_music_path)
                if bg_music.duration < final_video.duration:
                    repeat_times = int(final_video.duration / bg_music.duration) + 1
                    bg_music = mp.concatenate_audioclips([bg_music] * repeat_times)
                    bg_music = bg_music.subclipped(0, final_video.duration)
                elif bg_music.duration > final_video.duration:
                    bg_music = bg_music.subclipped(0, final_video.duration)
                bg_music = bg_music.with_volume_scaled(0.2)
                if final_video.audio:
                    tts_audio = final_video.audio.with_duration(
                        final_video.duration).with_volume_scaled(1.0)
                    bg_audio = bg_music.with_duration(
                        final_video.duration).with_volume_scaled(0.15)
                    mixed_audio = mp.CompositeAudioClip(
                        [tts_audio,
                         bg_audio]).with_duration(final_video.duration)
                else:
                    mixed_audio = bg_music.with_duration(
                        final_video.duration).with_volume_scaled(0.3)
                final_video = final_video.with_audio(mixed_audio)

        assert final_video is not None
        logger.info('Rendering final video...')
        logger.info(
            f'Total video duration: {final_video.duration:.1f} seconds')
        logger.info(f'Video resolution: {final_video.size}')
        logger.info(
            f"Audio status: {'Has audio' if final_video.audio else 'No audio'}"
        )
        logger.info(f'final_video type: {type(final_video)}')
        logger.info(f'final_video attributes: {dir(final_video)}')

        final_video.write_videofile(
            output_path,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            logger=None,
            threads=2,
            bitrate='5000k',
            audio_bitrate='192k',
            audio_fps=44100,
            write_logfile=False)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            test_clip = mp.VideoFileClip(output_path)
            actual_duration = test_clip.duration
            test_clip.close()
            if abs(actual_duration - final_video.duration) >= 1.0:
                raise RuntimeError('Duration not match')

    async def execute_code(self, inputs, **kwargs):
        messages, context = inputs
        final_name = 'preview_with_placeholders.mp4' if self.animation_mode == 'human' else 'final_video.mp4'
        final_video_path = os.path.join(self.work_dir, final_name)
        background_path = context.get('background_path')
        foreground_paths = context.get('foreground_paths')
        audio_paths = context.get('audio_paths')
        subtitle_paths = context.get('subtitle_paths')
        illustration_paths = context.get('illustration_paths')
        segments = context.get('segments')
        subtitle_segments_list = context.get('subtitle_segments_list')

        self.compose_final_video(
            background_path=background_path,
            foreground_paths=foreground_paths,
            audio_paths=audio_paths,
            subtitle_paths=subtitle_paths,
            illustration_paths=illustration_paths,
            segments=segments,
            output_path=final_video_path,
            subtitle_segments_list=subtitle_segments_list)
        return messages, context

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