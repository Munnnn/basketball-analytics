"""
Gradio application callbacks
"""

import os
import tempfile
import json
from typing import Optional, Tuple

from pipeline import VideoProcessor, ProcessorConfig
from config import get_settings, get_model_path
from utils import cleanup_resources, setup_logging


class VideoAnalysisCallbacks:
    """Handle Gradio UI callbacks"""
    
    def __init__(self):
        self.processor = None
        self.last_result = None
        self.temp_dir = tempfile.mkdtemp()
        
        # Setup logging
        setup_logging()
        
    def analyze_video(self,
                     video_path: str,
                     start_frame: int,
                     end_frame: Optional[int],
                     enable_pose: bool,
                     save_output: bool,
                     progress=gr.Progress()) -> Tuple[str, str, str, str]:
        """
        Analyze video callback
        
        Returns:
            Tuple of (output_video_path, analytics_json, timeline_html, status)
        """
        try:
            # Validate input
            if not video_path:
                return None, "", "", "❌ Please provide a video"
                
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
            self.processor = VideoProcessor(config)
            
            # Process video
            def progress_callback(p):
                progress(p, desc=f"Processing frames... {int(p*100)}%")
                
            self.last_result = self.processor.process_video(
                video_path, 
                progress_callback
            )
            
            # Generate outputs
            output_video = config.output_video_path if save_output else None
            
            analytics_json = json.dumps(
                self.last_result.to_dict(), 
                indent=2
            )
            
            timeline_html = self._generate_timeline_html(self.last_result)
            
            status = f"✅ Analysis complete! Processed {self.last_result.processed_frames} frames"
            
            return output_video, analytics_json, timeline_html, status
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
            return None, "", "", error_msg
            
        finally:
            cleanup_resources()
            
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
            # Implementation depends on requirements
            pass
            
        return output_path
        
    def _generate_timeline_html(self, result) -> str:
        """Generate HTML timeline visualization"""
        html = f"""
        <div class="timeline-container">
            <h3>Game Timeline</h3>
            <div class="stats">
                <p>Total Possessions: {len(result.possessions)}</p>
                <p>Total Events: {len(result.events)}</p>
                <p>Players Tracked: {len(result.tracks)}</p>
            </div>
            <div class="timeline">
                <!-- Timeline visualization would go here -->
            </div>
        </div>
        """
        return html
