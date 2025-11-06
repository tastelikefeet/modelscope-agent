import os
from dataclasses import dataclass, field
from typing import List

import edge_tts
from moviepy.editor import AudioClip, AudioFileClip
from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM
from ms_agent.llm.openai_llm import OpenAI
from ms_agent.utils import get_logger
from omegaconf import DictConfig, OmegaConf

logger = get_logger(__name__)


@dataclass
class Pattern:

    name: str
    pattern: str
    tags: List[str] = field(default_factory=list)


class GenerateAudio(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.llm: OpenAI = LLM.from_config(self.config)

    async def run(self, inputs, **kwargs):
        messages, context = inputs
        segments = context['segments']
        context['audio_paths'] = []
        tts_dir = os.path.join(self.work_dir, 'audio')
        os.makedirs(tts_dir, exist_ok=True)
        subtitle_dir = os.path.join(self.work_dir, 'subtitles')
        os.makedirs(subtitle_dir, exist_ok=True)

        for i, segment in enumerate(segments):
            audio_path = os.path.join(tts_dir, f'segment_{i + 1}.mp3')
            await self.generate_audio(segment, audio_path)
            context['audio_paths'].append(audio_path)
        return segments, context

    @staticmethod
    async def create_silent_audio(output_path, duration=5.0):
        import numpy as np

        def make_frame(t):
            return np.array([0.0, 0.0])

        audio = AudioClip(make_frame, duration=duration, fps=44100)
        audio.write_audiofile(output_path, verbose=False, logger=None)
        audio.close()

    @staticmethod
    async def edge_tts_generate(text, output_file, speaker='male'):

        text = text.strip()
        if not text:
            return False

        voices = OmegaConf.load(os.path.join(__file__, 'voices.yaml'))
        voice, params = voices.get(speaker, voices['male'])
        rate = params.get('rate', '+0%')
        pitch = params.get('pitch', '+0Hz')
        output_dir = os.path.dirname(output_file) or '.'
        os.makedirs(output_dir, exist_ok=True)
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
            return False

    @staticmethod
    def get_audio_duration(audio_path):
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        audio_clip.close()
        return duration

    async def generate_audio(self, segment, audio_path):
        tts_text = segment.get('content', '')
        if tts_text:
            if await self.edge_tts_generate(tts_text, audio_path):
                segment['audio_duration'] = self.get_audio_duration(audio_path)
            else:
                await self.create_silent_audio(audio_path, duration=3.0)
                segment['audio_duration'] = 3.0
        else:
            await self.create_silent_audio(audio_path, duration=2.0)
            segment['audio_duration'] = 2.0
