import re
import uuid
from dataclasses import dataclass, field
import json
import os
from typing import List, Union

from omegaconf import DictConfig, OmegaConf

from ms_agent.agent import CodeAgent
from ms_agent.llm import Message, LLM
from ms_agent.llm.openai_llm import OpenAI
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
        self.llm: OpenAI = LLM.from_config(self.config)


    async def run(self, inputs: Union[str, List[Message]],
                  **kwargs) -> List[Message]:
        return await self._generate_audio(inputs)

    @staticmethod
    async def create_silent_audio(output_path, duration=5.0):
        from moviepy.editor import AudioClip
        import numpy as np

        def make_frame(t):
            return np.array([0.0, 0.0])

        audio = AudioClip(make_frame, duration=duration, fps=44100)
        audio.write_audiofile(output_path, verbose=False, logger=None)
        audio.close()

    @staticmethod
    async def edge_tts_generate(text, output_file, speaker='male'):
        import edge_tts
        text = text.strip()
        if not text:
            return False

        voices = OmegaConf.load(os.path.join(__file__, 'voices.yaml'))
        voice, params = voices.get(speaker, voices['male'])
        rate = params.get('rate', '+0%')
        pitch = params.get('pitch', '+0Hz')
        output_dir = os.path.dirname(output_file) or '.'
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Using voice: {voice}, rate: {rate}, pitch: {pitch}')
        communicate = edge_tts.Communicate(
            text=text, voice=voice, rate=rate, pitch=pitch)

        audio_data = b''
        chunk_count = 0
        async for chunk in communicate.stream():
            if chunk['type'] == 'audio':
                audio_data += chunk['data']
                chunk_count += 1

        if len(audio_data) > 0:
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            return True
        else:
            print('No audio data received.')
            return False

    @staticmethod
    def get_audio_duration(audio_path):
        from moviepy.editor import AudioFileClip
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        audio_clip.close()
        return duration

    async def generate_audio(self, segment, audio_path):
        tts_text = segment.get('content', '')

        # Generate TTS
        if tts_text:
            generated = await self.edge_tts_generate(tts_text, audio_path)
            if generated:
                segment[
                    'audio_duration'] = self.get_audio_duration(
                    audio_path)
            else:
                await self.create_silent_audio(audio_path, duration=3.0)
                segment['audio_duration'] = 3.0
        else:
            await self.create_silent_audio(audio_path, duration=2.0)
            segment['audio_duration'] = 2.0

    async def _generate_audio(self, messages: List[Message]) -> str:
        segments = messages

        # 2. Generate assets for each segment
        asset_paths = {
            'audio_paths': [],
            'foreground_paths': [],
            'subtitle_paths': [],
            'illustration_paths': [],
            'subtitle_segments_list': []
        }

        full_output_dir = self.work_dir

        tts_dir = os.path.join(full_output_dir, 'audio')
        os.makedirs(tts_dir, exist_ok=True)

        subtitle_dir = os.path.join(full_output_dir, 'subtitles')
        os.makedirs(subtitle_dir, exist_ok=True)

        for i, segment in enumerate(segments):
            logger.info(
                f"[video_agent] Processing segment {i+1}/{len(segments)}: {segment['type']}"
            )
            audio_path = os.path.join(tts_dir, f'segment_{i + 1}.mp3')
            await self.generate_audio(segment, audio_path)
            asset_paths['audio_paths'].append(audio_path)
        return segments, asset_paths
