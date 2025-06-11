"""
Enhanced Gradio application callbacks with file path support
"""

import os
import gradio as gr
import tempfile
import json
from typing import Optional, Tuple
from pathlib import Path

from pipeline import VideoProcessor, ProcessorConfig
from config import get_settings, get_model_path
from utils import cleanup_resources, setup_logging


class VideoAnalysisCallbacks:
    """Handle Gradio UI callbacks with file path support"""

    def __init__(self):
        self.processor = None
        self.last_result = None
        self.temp_dir = tempfile.mkdtemp()

        # Setup logging
        setup_logging()

    def analyze_video(self,
                     input_method: str,
                     video_upload: Optional[str],
                     video_path: str,
                     start_frame: int,
                     end_frame: Optional[int],
                     enable_pose: bool,
                     save_output: bool,
                     progress=gr.Progress()) -> Tuple[str, str, str, str]:
        """
        Analyze video callback supporting both upload and file path

        Args:
            input_method: "Upload File" or "File Path"
            video_upload: Uploaded video file path (from Gradio)
            video_path: Direct file path string
            start_frame: Starting frame number
            end_frame: Ending frame number (0 for all)
            enable_pose: Enable pose estimation
            save_output: Save output video
            progress: Gradio progress callback

        Returns:
            Tuple of (output_video_path, analytics_json, timeline_html, status)
        """
        try:
            # Determine actual video path based on input method
            actual_video_path = self._get_video_path(input_method, video_upload, video_path)
            
            if not actual_video_path:
                return None, "", "", "❌ Please provide a valid video file or path"

            # Validate the video file exists and is accessible
            if not os.path.exists(actual_video_path):
                return None, "", "", f"❌ Video file not found: {actual_video_path}"

            # Get file size for progress estimation
            file_size_mb = os.path.getsize(actual_video_path) / (1024 * 1024)
            
            # Update status
            progress(0.05, desc="Initializing...")

            # Get settings
            settings = get_settings()

            # Create processor config
            config = ProcessorConfig(
                yolo_model_path=get_model_path('yolo'),
                sam_checkpoint_path=get_model_path('sam', 'edgetam'),
                start_frame=start_frame,
                end_frame=end_frame,
                enable_pose_estimation=enable_pose,
                enable_visualization=True,
                output_video_path=os.path.join(self.temp_dir, "output.mp4") if save_output else None,
                analytics_output_path=os.path.join(self.temp_dir, "analytics.json")
            )

            # Create processor
            progress(0.1, desc="Loading models...")
            self.processor = VideoProcessor(config)

            # Process video with enhanced progress tracking
            def progress_callback(p):
                # Map processing progress to 10-90% range
                mapped_progress = 0.1 + (p * 0.8)
                progress(mapped_progress, desc=f"Processing frames... {int(p*100)}%")

            progress(0.1, desc=f"Starting analysis of {file_size_mb:.1f}MB video...")
            
            self.last_result = self.processor.process_video(
                actual_video_path,
                progress_callback
            )

            # Generate outputs
            progress(0.95, desc="Generating outputs...")
            
            output_video = config.output_video_path if save_output else None

            analytics_json = json.dumps(
                self.last_result.to_dict(),
                indent=2
            )

            timeline_html = self._generate_timeline_html(self.last_result)

            status = self._generate_analysis_summary(self.last_result, file_size_mb)

            progress(1.0, desc="Complete!")

            return output_video, analytics_json, timeline_html, status

        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return None, "", "", error_msg

        finally:
            cleanup_resources()

    def _get_video_path(self, input_method: str, video_upload: Optional[str], video_path: str) -> Optional[str]:
        """Get the actual video path based on input method"""
        if input_method == "Upload File":
            return video_upload
        elif input_method == "File Path":
            return video_path.strip() if video_path else None
        return None

    def _generate_analysis_summary(self, result, file_size_mb: float) -> str:
        """Generate a comprehensive analysis summary"""
        summary = f"""✅ Analysis Complete!

📊 **Processing Summary:**
• Video Size: {file_size_mb:.1f} MB
• Frames Processed: {result.processed_frames}
• Players Tracked: {len(result.tracks)}
• Total Possessions: {len(result.possessions)}
• Events Detected: {len(result.events)}

🏀 **Team Analysis:**
• Teams Identified: {len(result.teams)}
• Player Assignments: {sum(len(team.players) for team in result.teams)}

⏱️ **Performance:**
• Processing completed successfully
• All outputs generated
"""
        return summary

    def validate_path(self, path: str) -> str:
        """Validate video file path"""
        if not path:
            return "❌ Please enter a file path"
        
        try:
            path_obj = Path(path)
            
            if not path_obj.exists():
                return f"❌ File does not exist: {path}"
            
            if not path_obj.is_file():
                return f"❌ Path is not a file: {path}"
            
            # Check if it's a video file by extension
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mp3'}
            if path_obj.suffix.lower() not in video_extensions:
                return f"⚠️ Warning: File extension '{path_obj.suffix}' may not be a video format"
            
            # Check file size
            size_mb = path_obj.stat().st_size / (1024 * 1024)
            
            return f"✅ Valid video file found ({size_mb:.1f} MB)"
            
        except Exception as e:
            return f"❌ Error accessing file: {str(e)}"

    def toggle_input_method(self, method: str) -> Tuple:
        """Toggle between upload and file path input methods"""
        if method == "Upload File":
            return (
                gr.update(visible=True),   # video_input
                gr.update(visible=False),  # video_path
                gr.update(visible=False),  # validate_path_btn
                gr.update(visible=False),  # path_status
            )
        else:  # File Path
            return (
                gr.update(visible=False),  # video_input
                gr.update(visible=True),   # video_path
                gr.update(visible=True),   # validate_path_btn
                gr.update(visible=True),   # path_status
            )

    def reset_system(self) -> str:
        """Reset system callback"""
        cleanup_resources()
        self.processor = None
        self.last_result = None
        return "✅ System reset complete"

    def export_analytics(self, format: str) -> str:
        """Export analytics in specified format"""
        if self.last_result is None:
            return None

        output_path = os.path.join(self.temp_dir, f"analytics.{format}")

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(self.last_result.to_dict(), f, indent=2)
        elif format == "csv":
            # Convert to CSV format
            import pandas as pd
            
            # Create a comprehensive CSV export
            data = []
            for frame_idx, frame_data in enumerate(self.last_result.frames):
                for detection in frame_data.detections:
                    data.append({
                        'frame': frame_idx,
                        'track_id': detection.track_id,
                        'class': detection.class_name,
                        'confidence': detection.confidence,
                        'x': detection.bbox[0],
                        'y': detection.bbox[1],
                        'width': detection.bbox[2],
                        'height': detection.bbox[3],
                        'team': detection.team_id if hasattr(detection, 'team_id') else None
                    })
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)

        return output_path

    def _generate_timeline_html(self, result) -> str:
        """Generate HTML timeline visualization"""
        # Create a more detailed timeline
        events_html = ""
        for i, event in enumerate(result.events[:10]):  # Show first 10 events
            events_html += f"""
            <div class="timeline-event">
                <span class="event-time">Frame {event.frame}</span>
                <span class="event-type">{event.type}</span>
                <span class="event-description">{event.description}</span>
            </div>
            """
        
        html = f"""
        <div class="timeline-container" style="font-family: Arial, sans-serif;">
            <h3>🏀 Game Timeline</h3>
            <div class="stats" style="background: #f0f0f0; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                    <div><strong>Total Possessions:</strong> {len(result.possessions)}</div>
                    <div><strong>Total Events:</strong> {len(result.events)}</div>
                    <div><strong>Players Tracked:</strong> {len(result.tracks)}</div>
                    <div><strong>Frames Processed:</strong> {result.processed_frames}</div>
                </div>
            </div>
            <div class="timeline" style="max-height: 400px; overflow-y: auto;">
                <h4>Recent Events:</h4>
                {events_html}
            </div>
        </div>
        <style>
        .timeline-event {{
            padding: 8px;
            margin: 5px 0;
            border-left: 3px solid #007bff;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .event-time {{
            font-weight: bold;
            color: #007bff;
            margin-right: 10px;
        }}
        .event-type {{
            background: #e9ecef;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.8em;
            margin-right: 10px;
        }}
        </style>
        """
        return html
