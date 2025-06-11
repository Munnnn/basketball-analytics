
"""
Main Gradio application
"""

import gradio as gr
from .components import create_ui_components
from .callbacks import VideoAnalysisCallbacks


def create_app():
    """Create Gradio application"""

    # Initialize callbacks
    callbacks = VideoAnalysisCallbacks()

    # Create UI
    with gr.Blocks(title="🏀 Basketball Analytics", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🏀 Basketball Analytics System

        Advanced AI-powered basketball analysis with player tracking, team identification,
        and play classification.
        """)

        # Create UI components
        components = create_ui_components()

        # Wire up callbacks
        components['analyze_btn'].click(
            fn=callbacks.analyze_video,
            inputs=[
                components['video_input'],
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

        components['reset_btn'].click(
            fn=callbacks.reset_system,
            outputs=[components['status_output']]
        )

        components['export_btn'].click(
            fn=callbacks.export_analytics,
            inputs=[components['export_format']],
            outputs=[components['download_output']]
        )

    return app


def launch_app(share: bool = False, port: int = 7860):
    """Launch Gradio application"""
    app = create_app()
    app.launch(share=share, server_port=port)


if __name__ == "__main__":
    launch_app(share=True)
