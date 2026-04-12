"""
Gradio UI components with file path support
"""

import gradio as gr


def create_ui_components():
    """Create all UI components"""
    components = {}
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input components
            with gr.Accordion("📹 Video Input", open=True):
                components['input_method'] = gr.Radio(
                    choices=["Upload File", "File Path"],
                    value="Upload File",
                    label="Input Method",
                    info="Choose to upload a file or provide a local file path"
                )
                
                components['video_input'] = gr.Video(
                    label="Upload Basketball Video",
                    visible=True
                )
                
                components['video_path'] = gr.Textbox(
                    label="Video File Path",
                    placeholder="Enter full path to video file (e.g., C:/Videos/basketball.mp4)",
                    visible=False,
                    info="For large files, use file path to avoid upload time"
                )
                
                components['validate_path_btn'] = gr.Button(
                    "✅ Validate Path",
                    size="sm",
                    visible=False
                )
                
                components['path_status'] = gr.Textbox(
                    label="Path Status",
                    visible=False,
                    interactive=False
                )
            
            with gr.Accordion("⚙️ Settings", open=True):
                with gr.Row():
                    components['start_frame'] = gr.Number(
                        label="Start Frame",
                        value=0,
                        precision=0
                    )
                    components['end_frame'] = gr.Number(
                        label="End Frame (0=all)",
                        value=0,
                        precision=0
                    )
                    
                components['enable_pose'] = gr.Checkbox(
                    label="Enable Pose Estimation",
                    value=False
                )
                
                components['save_output'] = gr.Checkbox(
                    label="Save Output Video",
                    value=True
                )
                
            # Control buttons
            with gr.Row():
                components['analyze_btn'] = gr.Button(
                    "🎯 Analyze Video",
                    variant="primary",
                    size="lg"
                )
                components['reset_btn'] = gr.Button(
                    "🔄 Reset",
                    variant="stop",
                    size="sm"
                )
                
        with gr.Column(scale=2):
            # Output components
            components['video_output'] = gr.Video(
                label="Analysis Result",
                height=400
            )
            
            components['status_output'] = gr.Textbox(
                label="Status",
                lines=3,
                max_lines=10
            )
            
            with gr.Tabs():
                with gr.Tab("📊 Analytics"):
                    components['analytics_output'] = gr.Code(
                        label="Analysis Data",
                        language="json"
                    )
                    
                with gr.Tab("📈 Timeline"):
                    components['timeline_output'] = gr.HTML(
                        label="Game Timeline"
                    )
                    
                with gr.Tab("💾 Export"):
                    components['export_format'] = gr.Radio(
                        choices=["json", "csv"],
                        value="json",
                        label="Export Format"
                    )
                    components['export_btn'] = gr.Button("Export Analytics")
                    components['download_output'] = gr.File(
                        label="Download"
                    )
                    
    return components
