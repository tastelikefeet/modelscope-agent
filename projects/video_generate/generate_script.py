import os
import json
from typing import List, Union

from omegaconf import DictConfig
from ms_agent.agent import CodeAgent
from ms_agent.llm import Message
from projects.video_generate.core import workflow as video_workflow
from ms_agent.utils import get_logger

logger = get_logger()


class GenerateScript(CodeAgent):
    """Step: Generate script

    Generate a script for later animate video.
    """

    def __init__(self, config: DictConfig, tag: str, trust_remote_code: bool = False, **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        # work_dir for intermediates
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        os.makedirs(self.work_dir, exist_ok=True)
        self.meta_path = os.path.join(self.work_dir, 'meta.json')
        # animation mode: auto (default) or human (manual animation workflow)
        self.animation_mode = getattr(self.config, 'animation_mode', 'auto')

    @staticmethod
    def _safe_topic(name: str) -> str:
        """Create a folder based on the topic"""
        import re
        safe = re.sub(r'[^\w\u4e00-\u9fff\-_]', '_', name or 'topic')
        safe = safe[:50] if len(safe) > 50 else safe
        return safe or 'topic'

    async def run(self, inputs: Union[str, List[Message]], **kwargs):
        assert isinstance(inputs, str)
        topic = inputs

        logger.info(f"[video_agent] Generating script for topic: {topic}")
        script = video_workflow.generate_script(topic)
        topic_dir = os.path.join(self.work_dir, self._safe_topic(topic))
        os.makedirs(topic_dir, exist_ok=True)

        # Save the script to a file to pass to the next step
        script_path = os.path.join(topic_dir, "script.txt")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        # persist topic for later steps
        with open(os.path.join(topic_dir, 'meta.json'), 'w', encoding='utf-8') as mf:
            json.dump({"topic": topic}, mf, ensure_ascii=False, indent=2)
            
        return script_path
