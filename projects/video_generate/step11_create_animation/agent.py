from ms_agent.agent.base import Agent
from ms_agent.utils import get_logger

logger = get_logger()


class HumanAnimation(Agent):

    async def run(self, inputs, **kwargs):
        logger.warn('HumanAnimationMode not supported for now.')
        # TODO
        return inputs
