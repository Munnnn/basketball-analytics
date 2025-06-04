"""
REST API implementation
"""

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from typing import Optional

from pipeline import VideoProcessor, ProcessorConfig
from config import get_settings, get_model_path
from utils import cleanup_resources


def create_api():
    """Create Flask REST API"""
    app = Flask(__name__)
    
    # Configure upload folder
    app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({'status': 'healthy'})
    
    @app.route('/analyze', methods=['POST'])
    def analyze_video():
        """Analyze video endpoint"""
        try:
            # Check if video file is present
            if 'video' not in request.files:
                return jsonify({'error': 'No video file provided'}), 400
            
            video_file = request.files['video']
            if video_file.filename == '':
                return jsonify({'error': 'No video file selected'}), 400
            
            # Save uploaded file
            filename = secure_filename(video_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(filepath)
            
            # Get parameters from request
            start_frame = request.form.get('start_frame', 0, type=int)
            end_frame = request.form.get('end_frame', None, type=int)
            enable_pose = request.form.get('enable_pose', False, type=bool)
            
            # Create processor config
            config = ProcessorConfig(
                yolo_model_path=get_model_path('yolo'),
                start_frame=start_frame,
                end_frame=end_frame,
                enable_pose_estimation=enable_pose,
                enable_visualization=False,  # No visualization for API
                save_analytics=True
            )
            
            # Process video
            processor = VideoProcessor(config)
            result = processor.process_video(filepath)
            
            # Clean up
            os.remove(filepath)
            cleanup_resources()
            
            # Return results
            return jsonify({
                'success': True,
                'data': result.to_dict()
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/models', methods=['GET'])
    def list_models():
        """List available models"""
        from config import ModelPaths
        model_paths = ModelPaths()
        return jsonify(model_paths.list_models())
    
    return app


def run_api(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Run REST API server"""
    app = create_api()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_api()
