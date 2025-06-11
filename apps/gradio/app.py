"""
Enhanced Gradio application with file path support
"""

import gradio as gr
from .components import create_ui_components
from .callbacks import VideoAnalysisCallbacks


def create_app():
    """Create enhanced Gradio application with file path support"""

    # Initialize callbacks
    callbacks = VideoAnalysisCallbacks()

    # Create UI
    with gr.Blocks(title="🏀 Basketball Analytics", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🏀 Basketball Analytics System

        Advanced AI-powered basketball analysis with player tracking, team identification,
        and play classification.
        
        **💡 Tip for Large Files:** Use "File Path" input method to avoid upload time for local videos (2GB+ files).
        """)

        # Create UI components
        components = create_ui_components()

        # Wire up input method toggle
        components['input_method'].change(
            fn=callbacks.toggle_input_method,
            inputs=[components['input_method']],
            outputs=[
                components['video_input'],
                components['video_path'],
                components['validate_path_btn'],
                components['path_status']
            ]
        )

        # Wire up path validation
        components['validate_path_btn'].click(
            fn=callbacks.validate_path,
            inputs=[components['video_path']],
            outputs=[components['path_status']]
        )

        # Auto-validate path on change
        components['video_path'].change(
            fn=callbacks.validate_path,
            inputs=[components['video_path']],
            outputs=[components['path_status']]
        )

        # Wire up main analysis callback
        components['analyze_btn'].click(
            fn=callbacks.analyze_video,
            inputs=[
                components['input_method'],
                components['video_input'],
                components['video_path'],
                components['start_frame'],
                components['end_frame'],
                components['enable_pose'],
                components['save_output']
            ],
            outputs=[
                components['video_output'],
                components['analytics_output'],
                components['timeline_output'],
                components['status_output']
            ],
            show_progress=True
        )

        # Wire up reset functionality
        components['reset_btn'].click(
            fn=callbacks.reset_system,
            outputs=[components['status_output']]
        )

        # Wire up export functionality
        components['export_btn'].click(
            fn=callbacks.export_analytics,
            inputs=[components['export_format']],
            outputs=[components['download_output']]
        )

        # Add some helpful examples
        with gr.Accordion("📝 Usage Examples", open=False):
            gr.Markdown("""
            ### File Path Examples:
            
            **Windows:**
            - `C:\\Videos\\basketball_game.mp4`
            - `D:\\Sports\\NBA_Finals_2024.mp4`
            
            **macOS/Linux:**
            - `/Users/username/Videos/basketball.mp4`
            - `/home/user/Downloads/game_footage.mp4`
            
            ### Performance Tips:
            - Use **File Path** for files larger than 500MB to skip upload time
            - Enable **Pose Estimation** only when needed (increases processing time)
            - Set **Start/End Frame** to analyze specific segments
            - Large files (2GB+) process much faster with direct file paths
            """)

    return app


def launch_app(share: bool = False, port: int = 7860, debug: bool = False):
    """Launch enhanced Gradio application"""
    app = create_app()
    app.launch(
        share=share, 
        server_port=port,
        debug=debug,
        show_error=True
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch Basketball Analytics App')
    parser.add_argument('--share', action='store_true', help='Create shareable link')
    parser.add_argument('--port', type=int, default=7860, help='Port to run on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    launch_app(share=args.share, port=args.port, debug=args.debug)
