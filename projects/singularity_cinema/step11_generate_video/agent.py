# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import aiohttp
from omegaconf import DictConfig

from ms_agent.agent import CodeAgent
from ms_agent.llm import Message
from ms_agent.utils import get_logger

logger = get_logger()


class GenerateVideo(CodeAgent):

    def __init__(self,
                 config: DictConfig,
                 tag: str,
                 trust_remote_code: bool = False,
                 **kwargs):
        super().__init__(config, tag, trust_remote_code, **kwargs)
        self.work_dir = getattr(self.config, 'output_dir', 'output')
        self.num_parallel = getattr(self.config, 't2v_num_parallel', 1)
        self.video_prompts_dir = os.path.join(self.work_dir,
                                                     'video_prompts')
        self.videos_dir = os.path.join(self.work_dir, 'videos')
        os.makedirs(self.videos_dir, exist_ok=True)

    async def execute_code(self, messages: Union[str, List[Message]],
                           **kwargs) -> List[Message]:
        with open(os.path.join(self.work_dir, 'segments.txt'), 'r') as f:
            segments = json.load(f)
        video_prompts = []
        for i in range(len(segments)):
            if 'video' in segments[i]:
                with open(
                        os.path.join(self.video_prompts_dir,
                                     f'segment_{i + 1}.txt'), 'r') as f:
                    video_prompts.append(f.read())
            else:
                video_prompts.append(None)
        logger.info('Generating videos.')

        tasks = [
            (i, segment, prompt)
            for i, (segment,
                    prompt) in enumerate(zip(segments, video_prompts))
        ]

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = [
                executor.submit(self._process_single_video_static,
                               i, segment, prompt, self.config,
                               self.videos_dir)
                for i, segment, prompt in tasks
            ]
            # Wait for all tasks to complete
            for future in futures:
                future.result()

        return messages

    @staticmethod
    def _process_single_video_static(i, segment, prompt, config,
                                            videos_dir):
        """Static method for thread pool execution of video generation"""
        import asyncio
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                GenerateVideo._process_single_video_impl(
                    i, segment, prompt, config, videos_dir))
        finally:
            loop.close()

    @staticmethod
    async def _process_single_video_impl(i, segment, prompt, config,
                                          videos_dir):
        """Implementation of single video processing using OpenAI Sora API"""
        if prompt is None:
            logger.info(f'Skipping video generation for segment {i + 1} (no video prompt).')
            return

        output_path = os.path.join(videos_dir, f'video_{i + 1}.mp4')
        if os.path.exists(output_path):
            logger.info(f'Video already exists for segment {i + 1}: {output_path}')
            return

        logger.info(f'Generating video for segment {i + 1}: {prompt}')

        # Extract configuration
        base_url = config.text2video.t2v_base_url.strip('/')
        api_key = config.text2video.t2v_api_key
        model = config.text2video.t2v_model
        assert api_key is not None, "Video generation API key is required"

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

        # Prepare request payload for video generation
        payload = {
            'model': model,
            'prompt': prompt,
            'size': '1920x1080',  # Full HD video
        }

        async with aiohttp.ClientSession() as session:
            try:
                # Create video generation task
                async with session.post(
                        f'{base_url}/v1/videos/generations',
                        headers={**headers, 'X-DashScope-Async': 'enable'},
                        json=payload) as resp:
                    resp.raise_for_status()
                    response_data = await resp.json()
                    
                    # Check if response contains task_id (async mode)
                    if 'task_id' in response_data:
                        task_id = response_data['task_id']
                        logger.info(f'Video generation task created: {task_id}')
                        
                        # Poll for task completion
                        video_url = await GenerateVideo._poll_video_task(
                            session, base_url, task_id, headers)
                    elif 'output' in response_data and 'video_url' in response_data['output']:
                        # Synchronous response
                        video_url = response_data['output']['video_url']
                    else:
                        raise RuntimeError(f'Unexpected response format: {response_data}')

                # Download the generated video
                logger.info(f'Downloading video from: {video_url}')
                async with session.get(video_url) as video_resp:
                    video_resp.raise_for_status()
                    video_content = await video_resp.read()
                    with open(output_path, 'wb') as f:
                        f.write(video_content)
                    logger.info(f'Video saved to: {output_path}')

            except Exception as e:
                logger.error(f'Failed to generate video for segment {i + 1}: {str(e)}')
                raise

    @staticmethod
    async def _poll_video_task(session, base_url, task_id, headers):
        """Poll the video generation task until completion"""
        max_wait_time = 1800  # 30 minutes for video generation
        poll_interval = 5
        max_poll_interval = 30
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            await asyncio.sleep(poll_interval)
            elapsed_time += poll_interval

            async with session.get(
                    f'{base_url}/v1/tasks/{task_id}',
                    headers={**headers, 'X-DashScope-Task-Type': 'video_generation'}) as result:
                result.raise_for_status()
                data = await result.json()

                task_status = data.get('task_status') or data.get('status')
                logger.info(f'Task {task_id} status: {task_status}')

                if task_status in ['SUCCEEDED', 'SUCCEED', 'completed']:
                    # Extract video URL from response
                    if 'output' in data:
                        if isinstance(data['output'], dict):
                            video_url = data['output'].get('video_url') or data['output'].get('url')
                        elif isinstance(data['output'], str):
                            video_url = data['output']
                        else:
                            video_url = data['output'][0] if isinstance(data['output'], list) else None
                    elif 'result' in data:
                        video_url = data['result'].get('video_url') or data['result'].get('url')
                    else:
                        raise RuntimeError(f'Cannot find video URL in response: {data}')
                    
                    if not video_url:
                        raise RuntimeError(f'Video URL is empty in response: {data}')
                    
                    return video_url

                elif task_status in ['FAILED', 'failed', 'error']:
                    error_msg = data.get('error') or data.get('message') or 'Unknown error'
                    raise RuntimeError(f'Video generation failed: {error_msg}')

            # Exponential backoff for polling interval
            poll_interval = min(poll_interval * 1.2, max_poll_interval)

        raise TimeoutError(f'Video generation task {task_id} timed out after {max_wait_time} seconds')
