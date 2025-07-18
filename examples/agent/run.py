import asyncio
import os
import sys

from ms_agent import LLMAgent
from ms_agent.config import Config

path = os.path.dirname(os.path.abspath(__file__))
# ms_agent/agent/agent.yaml
# The system of the config will help LLM to make a better analysis and plan
agent_config = os.path.join(path, '..', '..', 'ms_agent', 'agent', 'agent.yaml')
# TODO change the mcp.json to a real path:
# https://www.modelscope.cn/mcp/servers/@amap/amap-maps
mcp_config = os.path.join(path, 'mcp.json')


async def run_query(query: str):
    config = Config.from_task(agent_config)
    # TODO change to your real api key
    config.llm.modelscope_api_key = '<your-modelscope-api-here>'
    engine = LLMAgent(
        config=config,
        mcp_server_file=mcp_config)

    _content = ''
    generator = await engine.run(query, stream=True)
    async for _response_message in generator:
        new_content = _response_message[-1].content[len(_content):]
        sys.stdout.write(new_content)
        sys.stdout.flush()
        _content = _response_message[-1].content
    sys.stdout.write('\n')
    return _content


if __name__ == '__main__':
    query = '帮我找一下杭州西湖附近的咖啡厅'
    asyncio.run(run_query(query))