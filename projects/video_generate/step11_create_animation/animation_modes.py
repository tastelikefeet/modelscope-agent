import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont
import subprocess
import json


class AnimationProductionMode(Enum):
    AUTO = 'auto'
    HUMAN_CONTROLLED = 'human'


class AnimationStatus(Enum):
    PENDING = 'pending'  # Waiting for production
    DRAFT = 'draft'  # Draft stage
    PREVIEW = 'preview'  # Preview stage
    REVISION = 'revision'  # Under revision
    APPROVED = 'approved'  # Approved
    COMPLETED = 'completed'  # Production completed
    FAILED = 'failed'  # Production failed


@dataclass
class AnimationTask:
    """Animation task data structure"""
    task_id: str
    segment_index: int
    content: str
    content_type: str
    mode: AnimationProductionMode
    status: AnimationStatus

    # Production related
    script: Optional[str] = None
    manim_code: Optional[str] = None
    preview_video_path: Optional[str] = None
    final_video_path: Optional[str] = None
    placeholder_path: Optional[str] = None

    # Human-machine interaction
    human_feedback: List[str] = None
    revision_count: int = 0
    max_revisions: int = 5

    # Time information
    audio_duration: float = 8.0
    creation_time: Optional[str] = None
    completion_time: Optional[str] = None

    def __post_init__(self):
        if self.human_feedback is None:
            self.human_feedback = []


@dataclass
class PlaceholderConfig:
    """Placeholder configuration"""
    width: int = 1280
    height: int = 720
    background_color: str = '#f0f0f0'
    text_color: str = '#333333'
    font_size: int = 48
    placeholder_text: str = 'Animation in production...'
    show_content_preview: bool = True
    show_progress_indicator: bool = True


class AnimationTaskManager:
    """Animation task management"""

    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.tasks_file = os.path.join(project_dir, 'animation_tasks.json')
        self.tasks: Dict[str, AnimationTask] = {}
        self.load_tasks()

    def create_task(self, segment_index, content, content_type, mode,
                    audio_duration):
        """Create new animation task, return ID directly if duplicate task exists"""
        import uuid
        from datetime import datetime

        # Check if a task already exists for the same segment
        existing_task = self.get_task_by_segment(segment_index, content_type)
        if existing_task:
            print(f'Found existing task: {existing_task.task_id}')
            return existing_task.task_id

        task_id = f'anim_{segment_index}_{uuid.uuid4().hex[:8]}'

        task = AnimationTask(
            task_id=task_id,
            segment_index=segment_index,
            content=content,
            content_type=content_type,
            mode=mode,
            status=AnimationStatus.PENDING,
            audio_duration=audio_duration,
            creation_time=datetime.now().isoformat())

        self.tasks[task_id] = task
        self.save_tasks()
        print(f'Created new task: {task_id}')
        return task_id

    def get_task_by_segment(self, segment_index, content_type):
        """Find task by segment index and content type"""
        for task in self.tasks.values():
            if task.segment_index == segment_index and task.content_type == content_type:
                return task
        return None

    def update_task_status(self, task_id, status):
        """Update task status"""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self.save_tasks()

    def add_human_feedback(self, task_id, feedback):
        """Add human feedback"""
        if task_id in self.tasks:
            self.tasks[task_id].human_feedback.append(feedback)
            self.tasks[task_id].revision_count += 1
            self.save_tasks()

    def get_task(self, task_id):
        """Get task"""
        return self.tasks.get(task_id)

    def get_tasks_by_status(self, status):
        """Get task list by status"""
        return [task for task in self.tasks.values() if task.status == status]

    def save_tasks(self):
        """Save tasks to file"""
        import json
        from dataclasses import asdict

        tasks_data = {}
        for task_id, task in self.tasks.items():
            task_dict = asdict(task)
            # Handle enum types
            task_dict['mode'] = task.mode.value
            task_dict['status'] = task.status.value
            tasks_data[task_id] = task_dict

        with open(self.tasks_file, 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, ensure_ascii=False, indent=2)

    def load_tasks(self):
        """Load tasks from file"""
        if not os.path.exists(self.tasks_file):
            return

        try:
            with open(self.tasks_file, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)

            for task_id, task_dict in tasks_data.items():
                # Restore enum types
                task_dict['mode'] = AnimationProductionMode(task_dict['mode'])
                task_dict['status'] = AnimationStatus(task_dict['status'])

                self.tasks[task_id] = AnimationTask(**task_dict)

        except Exception as e:
            print(f'Failed to load task file: {e}')


class PlaceholderGenerator:
    """Placeholder generation tool"""

    def __init__(self, config=None):
        self.config = config or PlaceholderConfig()

    def create_placeholder(self, task, output_path):
        """Create placeholder video"""
        img = Image.new('RGB', (self.config.width, self.config.height),
                        self.config.background_color)
        draw = ImageDraw.Draw(img)

        # Add placeholder text
        try:
            # Try to use custom font
            font_path = os.path.join(
                os.path.dirname(__file__), 'asset', '字魂龙吟手书(商用需授权).ttf')
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, self.config.font_size)
            else:
                font = ImageFont.load_default()
        except:  # noqa
            font = ImageFont.load_default()

        # Main title
        title = self.config.placeholder_text
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        title_x = (self.config.width - title_width) // 2
        title_y = self.config.height // 3

        draw.text((title_x, title_y),
                  title,
                  fill=self.config.text_color,
                  font=font)

        # Content preview
        if self.config.show_content_preview and task.content:
            content_preview = task.content[:50] + '...' if len(
                task.content) > 50 else task.content
            try:
                content_font = ImageFont.truetype(
                    font_path, self.config.font_size // 2) if os.path.exists(
                        font_path) else ImageFont.load_default()
            except:  # noqa
                content_font = ImageFont.load_default()

            content_bbox = draw.textbbox((0, 0),
                                         content_preview,
                                         font=content_font)
            content_width = content_bbox[2] - content_bbox[0]
            content_x = (self.config.width - content_width) // 2
            content_y = title_y + title_height + 50

            draw.text((content_x, content_y),
                      content_preview,
                      fill=self.config.text_color,
                      font=content_font)

        # Progress indicator
        if self.config.show_progress_indicator:
            status_text = f'Status: {task.status.value} | Type: {task.content_type}'
            try:
                status_font = ImageFont.truetype(
                    font_path, self.config.font_size // 3) if os.path.exists(
                        font_path) else ImageFont.load_default()
            except:  # noqa
                status_font = ImageFont.load_default()

            status_bbox = draw.textbbox((0, 0), status_text, font=status_font)
            status_width = status_bbox[2] - status_bbox[0]
            status_x = (self.config.width - status_width) // 2
            status_y = self.config.height - 100

            draw.text((status_x, status_y),
                      status_text,
                      fill=self.config.text_color,
                      font=status_font)

        # Save placeholder image
        temp_img_path = output_path.replace('.mov', '_placeholder.png')
        img.save(temp_img_path)

        # Convert to video
        try:
            cmd = [
                'ffmpeg', '-y', '-f', 'image2', '-loop', '1', '-i',
                temp_img_path, '-t',
                str(task.audio_duration), '-pix_fmt', 'yuv420p', '-r', '15',
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(temp_img_path)  # Clean up temporary file
            return output_path
        except Exception as e:
            print(f'Failed to create placeholder video: {e}')
            return None