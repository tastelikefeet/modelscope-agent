# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import base64
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from os import getcwd
from typing import List, Union

import json
from moviepy import VideoFileClip
from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.utils import get_logger
from omegaconf import DictConfig
from PIL import Image

logger = get_logger()


class GenerateVideo(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.num_parallel = getattr(self.config, 't2v_num_parallel', 1)
        self.video_prompts_dir = os.path.join(self.work_dir,
                                                     'video_prompts')
        self.videos_dir = os.path.join(self.work_dir, 'videos')
        os.makedirs(self.videos_dir, exist_ok=True)

    async def execute_code(self, messages: Union[str, List[Message]],
                           **kwargs) -> List[Message]:
        with open(os.path.join(self.work_dir, 'segments.txt'), 'r') as f:
            segments = json.load(f)
        video_prompts = []
        for i in range(len(segments)):
            if 'video' in segments[i]:
                with open(
                        os.path.join(self.video_prompts_dir,
                                     f'segment_{i + 1}.txt'), 'r') as f:
                    video_prompts.append(f.read())
            else:
                video_prompts.append(None)
        logger.info('Generating videos.')

        tasks = [
            (i, segment, prompt)
            for i, (segment,
                    prompt) in enumerate(zip(segments, video_prompts))
        ]

        # Use ThreadPoolExecutor with asyncio event loop
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = [
                loop.run_in_executor(executor,
                                     self._process_single_video_static,
                                     i, segment, prompt, self.config,
                                     self.videos_dir)
                for i, segment, prompt in tasks
            ]
            await asyncio.gather(*futures)

        return messages

    @staticmethod
    def _process_single_video_static(i, segment, prompt, config,
                                            images_dir):
        """Static method for multiprocessing"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                GenerateVideo._process_single_illustration_impl(
                    i, segment, prompt, config, images_dir))
        finally:
            loop.close()
