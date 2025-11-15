# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List

import edge_tts
import json
from moviepy import AudioClip, AudioFileClip
from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM
from ms_agent.llm.openai_llm import OpenAI
from ms_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger(__name__)


class ParseImages(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        _config = deepcopy(config)
        _config.llm = _config.mllm
        self.mllm: OpenAI = LLM.from_config(_config)

    async def execute_code(self, messages, **kwargs):
        logger.info('Parsing images.')
        docs_file = os.path.join(self.work_dir, 'docs.txt')
        if not os.path.exists(docs_file):
            return messages
        with open(docs_file, 'r') as f:
            docs = f.readlines()

        if not docs:
            return messages

        docs = [doc.strip() for doc in docs if doc.strip()]
        image_files = []
        for docs in docs:
            image_files.extend(self.parse_images(docs))

        for image_file in image_files:
            size = self.get_image_size(image_file)
            description = self.get_image_description(image_file)
        return messages

    def parse_images(self, filename):
        # TODO

    def get_image_size(self, filename):
        # TODO

    def get_image_description(self, filename):
        # TODO
