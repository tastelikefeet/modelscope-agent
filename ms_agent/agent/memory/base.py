# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import abstractmethod
from typing import List

from ..runtime import Runtime
from ms_agent.llm.utils import Message


class Memory:
    """The memory refine tool"""

    def __init__(self, config):
        self.config = config

    async def connect(self) -> None:
        """Connect to the Memory.

        Returns:
            None
        Raises:
            Exceptions if anything goes wrong.
        """
        pass

    async def cleanup(self) -> None:
        """Disconnect and clean up the memory.

        Returns:
            None
        Raises:
            Exceptions if anything goes wrong.
        """
        pass

    @abstractmethod
    async def run(self, runtime: Runtime, messages: List[Message]) -> List[Message]:
        """Refine the messages

        Args:
            runtime (Runtime): The runtime information of the current agent
            messages(`List[Message]`): The input messages

        Returns:
            The output messages
        """
        pass
