from omegaconf import DictConfig

from ms_agent import LLMAgent
from ms_agent.utils.constants import DEFAULT_TAG


class ArchitectureAgent(LLMAgent):

    def __init__(self,
                 config: DictConfig = DictConfig({}),
                 tag: str = DEFAULT_TAG,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
