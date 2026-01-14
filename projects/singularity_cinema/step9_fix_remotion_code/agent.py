# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union

import json
from ms_agent.agent import CodeAgent
from ms_agent.llm import LLM, Message
from ms_agent.utils import get_logger
from omegaconf import DictConfig

logger = get_logger()


class FixRemotionCode(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.num_parallel = getattr(self.config, 'llm_num_parallel', 10)
        self.code_fix_dir = os.path.join(self.work_dir, 'code_fix')
        os.makedirs(self.code_fix_dir, exist_ok=True)

    async def execute_code(self, messages: Union[str, List[Message]],
                           **kwargs) -> List[Message]:
        logger.info('Fixing remotion code.')
        with open(os.path.join(self.work_dir, 'segments.txt'), 'r') as f:
            segments = json.load(f)

        remotion_code_dir = os.path.join(self.work_dir, 'remotion_code')
        remotion_code = []
        pre_errors = []
        pre_error_mode = False
        for i in range(len(segments)):
            file_path = os.path.join(remotion_code_dir, f'Segment{i+1}.tsx')
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    remotion_code.append(f.read())
            else:
                remotion_code.append('')

            error_file = os.path.join(self.code_fix_dir,
                                      f'code_fix_{i + 1}.txt')
            if os.path.exists(error_file):
                pre_error_mode = True
                with open(error_file, 'r') as _f:
                    pre_error = _f.read()
                    pre_error = pre_error or ''
            else:
                pre_error = None
            pre_errors.append(pre_error)

        if pre_error_mode:
            pre_errors = [e or '' for e in pre_errors]
        else:
            pre_errors = [None] * len(segments)

        tasks = [
            (i, pre_error, code)
            for i, (code,
                    pre_error) in enumerate(zip(remotion_code, pre_errors))
            if code
        ]
        results = {}

        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = {
                executor.submit(self._process_single_code_static, i, pre_error,
                                code, self.config): i
                for i, pre_error, code in tasks
            }
            for future in as_completed(futures):
                i, code = future.result()
                results[i] = code

        final_results = [(i, results.get(i, '')) for i in range(len(segments))]

        if pre_error_mode:
            shutil.rmtree(self.code_fix_dir, ignore_errors=True)
        for (i, code) in final_results:
            if code:
                remotion_file = os.path.join(remotion_code_dir,
                                             f'Segment{i + 1}.tsx')
                with open(remotion_file, 'w', encoding='utf-8') as f:
                    f.write(code)

        return messages

    @staticmethod
    def _process_single_code_static(i, pre_error, code, config):
        """Static method for multiprocessing"""
        if not code:
            return i, ''

        llm = LLM.from_config(config)
        if pre_error is not None:
            logger.info(f'Try to fix pre defined error for segment {i+1}')
            if pre_error:
                logger.info(f'Fixing pre error of segment {i+1}: {pre_error}')
                code = FixRemotionCode._fix_code_impl(llm, pre_error, code)
                logger.info(f'Fix pre error of segment {i + 1} done')
        return i, code

    @staticmethod
    def _fix_code_impl(llm, fix_prompt, code):
        fix_request = f"""
{fix_prompt}

**原始代码**：
```
{code}
```

- 请专注于解决检测到的问题
- 保留正确的部分，只修复有问题的区域
- 确保不引入新的布局问题
- 在保持正确部分不变的前提下，进行最小化的代码修改来修复问题
- 输出必须是有效的 React 函数组件

**关键修复规则**：
1. **React 错误 #130（对象作为子元素）**：如果错误提示"Objects are not valid as a React child"，
   检查是否有变量被直接渲染（例如 `<div>{{style}}</div>`）。
   将其改为渲染属性（例如 `<div>{{style.width}}</div>`）。
2. **Remotion Interpolate 错误**：如果错误提示"outputRange must contain only numbers"，
   检查你的 `interpolate` 调用。
   确保 `outputRange` 具有一致的类型（全部为数字 或 全部为带相同单位的字符串）。
3. **黑屏 / 素材缺失**：如果错误提示"Visual Check Failed"或"Missing Assets"，请确保：
    - `opacity` 为 1。
    - `zIndex` 足够高。
    - 图片确实被使用了（`<Img src={{staticFile(...)}} />`）。

请精确修复检测到的问题。"""
        inputs = [Message(role='user', content=fix_request)]
        _response_message = llm.generate(inputs)
        response = _response_message.content

        # 使用正则表达式稳健地提取代码
        code_match = re.search(
            r'```(?:typescript|tsx|js|javascript)?\s*(.*?)```', response,
            re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code = response
            if 'import React' in code:
                idx = code.find('import React')
                code = code[idx:]

        return code.strip()

