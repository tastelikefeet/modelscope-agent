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
        """Implementation of single video processing supporting both OpenAI and DashScope APIs"""
        if prompt is None:
            logger.info(f'Skipping video generation for segment {i + 1} (no video prompt).')
            return

        output_path = os.path.join(videos_dir, f'video_{i + 1}.mp4')
        if os.path.exists(output_path):
            logger.info(f'Video already exists for segment {i + 1}: {output_path}')
            return

        logger.info(f'Generating video for segment {i + 1}: {prompt}')

        # Extract configuration
        api_key = config.text2video.t2v_api_key
        model = config.text2video.t2v_model
        provider = getattr(config.text2video, 't2v_provider', 'dashscope').lower()
        size = getattr(config.text2video, 't2v_size', '1280x720')
        work_dir = os.path.dirname(videos_dir)
        with open(os.path.join(work_dir, 'audio_info.txt'), 'r') as f:
            audio_infos = json.load(f)

        audio_duration = audio_infos[i]['audio_duration']
        fit_duration = config.text2video.t2v_seconds[0]
        for duration in config.text2video.t2v_seconds:
            fit_duration = duration
            if duration > audio_duration:
                break
        
        assert api_key is not None, "Video generation API key is required"
        logger.info(f'Using provider: {provider}')

        # Get provider-specific configuration
        provider_config = getattr(config.text2video, provider, None)
        if provider_config is None:
            raise ValueError(f'No configuration found for provider: {provider}')

        # Generate video using unified method
        video_url = await GenerateVideo._generate_video(
            provider_config, api_key, model, prompt, size, fit_duration)

        # Download the generated video
        logger.info(f'Downloading video from: {video_url}')
        async with aiohttp.ClientSession() as session:
            # Add auth header for OpenAI content endpoint
            headers = {}
            if video_url.startswith(provider_config.base_url) and hasattr(provider_config, 'content_endpoint'):
                headers['Authorization'] = f'Bearer {api_key}'
            async with session.get(video_url, headers=headers) as video_resp:
                video_resp.raise_for_status()
                video_content = await video_resp.read()
                with open(output_path, 'wb') as f:
                    f.write(video_content)
                logger.info(f'Video saved to: {output_path}')

    @staticmethod
    def _get_nested_value(data, path):
        """Get value from nested dict using path list"""
        value = data
        for key in path:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
            if value is None:
                return None
        return value

    @staticmethod
    def _build_request_payload(provider_config, model, prompt, size, seconds):
        """Build request payload based on provider format"""
        request_format = getattr(provider_config, 'request_format', 'openai')
        
        if request_format == 'dashscope':
            return {
                'model': model,
                'input': {
                    'prompt': prompt,
                    'size': size,
                    'seconds': seconds,
                },
                'parameters': {}
            }
        else:  # openai format
            return {
                'model': model,
                'prompt': prompt,
                'size': size,
                'seconds': str(seconds),
            }

    @staticmethod
    async def _generate_video(provider_config, api_key, model, prompt, size, seconds):
        """Unified video generation method for all providers"""
        base_url = provider_config.base_url.strip('/')
        create_endpoint = provider_config.create_endpoint
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        
        # Add async header if configured
        if hasattr(provider_config, 'async_header') and provider_config.async_header:
            headers[provider_config.async_header] = 'enable'

        payload = GenerateVideo._build_request_payload(
            provider_config, model, prompt, size, seconds)

        async with aiohttp.ClientSession() as session:
            try:
                # Create video generation task/job
                async with session.post(
                        f'{base_url}{create_endpoint}',
                        headers=headers,
                        json=payload) as resp:
                    resp.raise_for_status()
                    response_data = await resp.json()
                    
                    # Extract task/video ID using configured path
                    task_id = GenerateVideo._get_nested_value(
                        response_data, provider_config.task_id_path)
                    
                    if not task_id:
                        raise RuntimeError(f'No task ID in response: {response_data}')
                    
                    logger.info(f'Video generation task created: {task_id}')
                    
                    # Poll for completion
                    return await GenerateVideo._poll_video_task(
                        session, provider_config, task_id, headers, api_key)

            except Exception as e:
                logger.error(f'Failed to generate video: {str(e)}')
                raise

    @staticmethod
    async def _poll_video_task(session, provider_config, task_id, headers, api_key):
        """Unified polling method for all providers"""
        max_wait_time = 1800  # 30 minutes
        poll_interval = 5
        max_poll_interval = 30
        elapsed_time = 0
        
        base_url = provider_config.base_url.strip('/')
        poll_endpoint = provider_config.poll_endpoint.replace('{task_id}', task_id).replace('{video_id}', task_id)
        success_statuses = provider_config.success_status
        failed_statuses = provider_config.failed_status

        while elapsed_time < max_wait_time:
            await asyncio.sleep(poll_interval)
            elapsed_time += poll_interval

            async with session.get(
                    f'{base_url}{poll_endpoint}',
                    headers=headers) as result:
                result.raise_for_status()
                data = await result.json()

                # Extract status using configured path
                status = GenerateVideo._get_nested_value(data, provider_config.status_path)
                logger.info(f'Task {task_id} status: {status}, defailed message: {str(data)}')

                if status in success_statuses:
                    # Check if provider uses content endpoint (like OpenAI)
                    if hasattr(provider_config, 'content_endpoint') and provider_config.content_endpoint:
                        content_endpoint = provider_config.content_endpoint.replace('{video_id}', task_id)
                        return f'{base_url}{content_endpoint}'
                    
                    # Otherwise extract video URL from response
                    video_url = GenerateVideo._get_nested_value(data, provider_config.video_url_path)
                    if not video_url:
                        raise RuntimeError(f'Video URL not found in response: {data}')
                    return video_url

                elif status in failed_statuses:
                    error_msg = data['output'].get('message') or 'Unknown error'
                    raise RuntimeError(f'Video generation failed: {error_msg}')

            # Exponential backoff for polling interval
            poll_interval = min(poll_interval * 1.2, max_poll_interval)

        raise TimeoutError(f'Video generation task {task_id} timed out after {max_wait_time} seconds')
