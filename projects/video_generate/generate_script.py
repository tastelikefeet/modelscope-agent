import os
import json
from typing import List, Union

from omegaconf import DictConfig
from ms_agent.agent import CodeAgent
from ms_agent.llm import Message
from ms_agent.utils import get_logger

logger = get_logger()


class GenerateScript(LLMAgent):
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

    @staticmethod
    def generate_script(topic):
        prompt = f""""""  # noqa

        script = modai_model_request(prompt, max_tokens=1200, temperature=0.7)
        script = clean_script_content(script) if script else ''

        # 结尾检查和修复
        if script and not script.strip().endswith(('！', '。', '？')):
            fix_prompt = f"""请为以下AI科普短视频文案补全一个完整结尾，保持轻松幽默的风格。

    原文案：
    {script}

    请直接输出补全的结尾部分，不要重复原文，不要包含任何思考过程或说明："""
            fix = modai_model_request(fix_prompt, max_tokens=512, temperature=0.5)
            if fix and fix.strip():
                script = script.strip() + ' ' + fix.strip()
                script = clean_script_content(script)  # 再次清理
            else:
                print('结尾修复失败，使用原文案')

        return script.strip() if script else ''

    async def run(self, inputs: Union[str, List[Message]], **kwargs):
        assert isinstance(inputs, str)
        topic = inputs

        logger.info(f"[video_agent] Generating script for topic: {topic}")
        script = self.generate_script(topic)


        # Save the script to a file to pass to the next step
        script_path = os.path.join(topic_dir, "script.txt")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        # persist topic for later steps
        with open(os.path.join(topic_dir, 'meta.json'), 'w', encoding='utf-8') as mf:
            json.dump({"topic": topic}, mf, ensure_ascii=False, indent=2)

        return script_path
