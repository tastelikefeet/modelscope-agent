# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import abstractmethod, ABC
from typing import Any, AsyncGenerator, List, Optional, Union

from omegaconf import DictConfig

from ms_agent.llm import Message


class Agent(ABC):
    """
    Base class for all agents. Make sure your custom agents are derived from this class.
    Args:
        config (DictConfig): Pre-loaded configuration object.
    """

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        self.config = config
        self.tag = tag
        self.trust_remote_code = trust_remote_code

    @abstractmethod
    async def run(
            self, inputs: Union[str, List[Message]]
    ) -> Union[List[Message], AsyncGenerator[List[Message], Any]]:
        """
        Main method to execute the agent.

        This method should define the logic of how the agent processes input and generates output messages.

        Args:
            inputs (Union[str, List[Message]]): Input data for the agent. Can be a raw string prompt,
                                                or a list of previous interaction messages.
        Returns:
            List[Message]: A list of message objects representing the agent's response or interaction history.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()
