import os
from typing import List, Union
from ms_agent.agent.base import Agent
from ms_agent.llm import Message
from omegaconf import DictConfig
from projects.video_generate.core import workflow as video_workflow
from .human_animation_studio import AnimationStudio


class CreateBackground(Agent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.animation_mode = getattr(self.config, 'animation_code', 'auto')
    
    def _animate(self, asset_info_path: str) -> Union[str, None]:
        # In human mode, auto-generate placeholder foreground clips for non-text segments
        if self.animation_mode == 'human':
            studio = AnimationStudio(
                self.work_dir, workflow_instance=video_workflow)
            task_manager = studio.task_manager
            placeholder_gen = studio.placeholder_generator
            fg = asset_paths.get('foreground_paths', [])
            for i, seg in enumerate(segments):
                if seg.get('type') != 'text' and (i >= len(fg)
                                                  or not fg[i]):
                    audio_duration = seg.get('audio_duration', 8.0)
                    task_id = task_manager.create_task(
                        segment_index=i + 1,
                        content=seg.get('content', ''),
                        content_type=seg.get('type'),
                        mode=AnimationProductionMode.HUMAN_CONTROLLED,
                        audio_duration=audio_duration)
                    task = task_manager.get_task(task_id)
                    placeholder_path = os.path.join(
                        full_output_dir, f'scene_{i+1}_placeholder.mov')
                    placeholder_video = placeholder_gen.create_placeholder(
                        task, placeholder_path)
                    # ensure list capacity
                    while len(asset_paths['foreground_paths']) <= i:
                        asset_paths['foreground_paths'].append(None)
                    asset_paths['foreground_paths'][
                        i] = placeholder_video if placeholder_video else None

    async def run(self, inputs: Union[str, List[Message]],
                  **kwargs) -> List[Message]:
        self._animate(inputs)
