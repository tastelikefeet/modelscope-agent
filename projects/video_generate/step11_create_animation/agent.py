from ms_agent.agent import CodeAgent

from ms_agent.utils import get_logger

logger = get_logger()


class HumanAnimation(CodeAgent):

    async def execute_code(self, inputs, **kwargs):
        logger.warn('HumanAnimationMode not supported for now.')
        # TODO
        return inputs

    def save_history(self, messages, **kwargs):
        messages, context = messages
        self.config.context = context
        return super().save_history(messages, **kwargs)

    def read_history(self, messages, **kwargs):
        _config, _messages = super().read_history(messages, **kwargs)
        if _config is not None:
            context = _config['context']
            return _config, (_messages, context)
        else:
            return _config, _messages