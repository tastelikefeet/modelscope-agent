import os
from typing import List, Union

import json
from ms_agent.agent.base import Agent
from ms_agent.llm import Message
from omegaconf import DictConfig
from projects.video_generate.core import workflow as video_workflow


class CreateBackground(Agent):
    """A thin wrapper that dispatches to original workflow functions.
    It preserves all original prompts/logic. We only adapt to ms-agent's CodeAgent loading via code_file.
    """

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        # work_dir for intermediates
        self.work_dir = os.path.join(
            self.config.local_dir, 'output') if getattr(
                self.config, 'local_dir', None) else os.getcwd()
        os.makedirs(self.work_dir, exist_ok=True)
        self.meta_path = os.path.join(self.work_dir, 'meta.json')
        # animation mode: auto (default) or human (manual animation workflow)
        import os as _os
        self.animation_mode = _os.environ.get('MS_ANIMATION_MODE',
                                              'auto').strip().lower() or 'auto'
        print(f'[video_agent] Animation mode: {self.animation_mode}')
    
    def _synthesize_video(self, asset_info_path: str) -> Union[str, None]:
        # In human mode, auto-generate placeholder foreground clips for non-text segments
        if self.animation_mode == 'human':
            try:
                from projects.video_generate.core.human_animation_studio import AnimationStudio
                from projects.video_generate.core.animation_production_modes import AnimationProductionMode
                print(
                    '[video_agent] Human mode: generating placeholder clips for non-text segments...'
                )
                studio = AnimationStudio(
                    full_output_dir, workflow_instance=video_workflow)
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
                        if placeholder_video:
                            print(
                                f'[video_agent] Placeholder generated for segment {i+1}: {placeholder_video}'
                            )
                        else:
                            print(
                                f'[video_agent] Placeholder generation failed for segment {i+1}'
                            )
            except Exception as e:
                print(
                    f'[video_agent] Human mode placeholder generation failed: {e}'
                )

        # Compose final video (human mode generates a preview with placeholders)
        final_name = 'preview_with_placeholders.mp4' if self.animation_mode == 'human' else 'final_video.mp4'
        final_video_path = os.path.join(full_output_dir, final_name)

        composed_path = video_workflow.compose_final_video(
            background_path=background_path,
            foreground_paths=asset_paths['foreground_paths'],
            audio_paths=asset_paths['audio_paths'],
            subtitle_paths=asset_paths['subtitle_paths'],
            illustration_paths=asset_paths['illustration_paths'],
            segments=segments,
            output_path=final_video_path,
            subtitle_segments_list=asset_paths['subtitle_segments_list'])

        if composed_path and os.path.exists(composed_path):
            print(
                f'[video_agent] Final video successfully composed at: {composed_path}'
            )
            return composed_path
        else:
            print('[video_agent] Final video composition failed.')
            return None

    async def run(self, inputs: Union[str, List[Message]],
                  **kwargs) -> List[Message]:
        """Dispatch by self.tag to keep ChainWorkflow simple.
        Inputs is the query string for first step, else pass file paths between steps via return messages.
        """
        # Normalize inputs
        if isinstance(inputs, list) and inputs and hasattr(
                inputs[0], 'content'):
            in_text = inputs[0].content
        else:
            in_text = inputs if isinstance(inputs, str) else ''

        result_path = None
        if self.tag == 'generate_script':
            topic = in_text
            result_path = self._generate_script(topic)
        elif self.tag == 'generate_assets':
            # inputs should be the path from previous step
            # fallback: if inputs looks like a path, use it; else assume default script.txt
            script_path = in_text if os.path.exists(in_text) else os.path.join(
                self.work_dir, 'script.txt')
            # We need topic; reuse the folder name or query text if available
            topic = os.path.basename(os.path.dirname(script_path)) or 'topic'
            result_path = self._generate_assets_from_script(script_path, topic)
        elif self.tag == 'synthesize_video':
            asset_info_path = in_text if os.path.exists(
                in_text) else os.path.join(self.work_dir, 'asset_info.json')
            result_path = self._synthesize_video(asset_info_path)
        else:
            print(f'[video_agent] Unknown tag: {self.tag}')

        # Return as a single Message list so next agent receives a text content
        out_text = result_path or ''
        return [Message(role='assistant', content=out_text)]
