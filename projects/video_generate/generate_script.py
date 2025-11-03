import os

from omegaconf import DictConfig

from ms_agent import LLMAgent
from ms_agent.utils import get_logger

logger = get_logger()


class GenerateScript(LLMAgent):

    def __init__(self, config: DictConfig, tag: str, trust_remote_code: bool = False, **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        # work_dir for intermediates
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        os.makedirs(self.work_dir, exist_ok=True)
