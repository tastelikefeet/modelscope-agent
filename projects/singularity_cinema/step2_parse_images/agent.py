# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List
from urllib.request import urlretrieve

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
        self.image_dir = os.path.join(self.work_dir, 'foreground_images')
        os.makedirs(self.image_dir, exist_ok=True)

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

        def process_image(image_file):
            size = self.get_image_size(image_file)
            description = self.get_image_description(image_file)
            return size, description

        with ThreadPoolExecutor(max_workers=4) as executor:
            output = list(executor.map(process_image, image_files))

        filename = os.path.join(self.work_dir, 'image_info.txt')
        with open(filename, 'w') as f:
            for img_tuple in output:
                image_json = {
                    'size': img_tuple[0],
                    'description': img_tuple[1],
                }
                f.write(json.dumps(image_json) + '\n')
        return messages

    def parse_images(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        image_pattern = r'!\[.*?\]\((.*?)\)'
        urls = re.findall(image_pattern, content)
        
        image_dir = os.path.join(self.work_dir, 'foreground_images')
        os.makedirs(image_dir, exist_ok=True)
        
        local_paths = []
        for url in urls:
            if url.startswith(('http://', 'https://')):
                ext = os.path.splitext(url)[1] or '.png'
                local_file = os.path.join(image_dir, f"{uuid.uuid4().hex[:8]}{ext}")
                urlretrieve(url, local_file)
                local_paths.append(local_file)
            else:
                local_paths.append(url)
        
        return local_paths

    def get_image_size(self, filename):
        # TODO

    def get_image_description(self, filename):
        # TODO
