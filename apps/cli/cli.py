"""
Command-line interface implementation
"""

import click
import json
from pathlib import Path

from pipeline import VideoProcessor, ProcessorConfig
from config import get_settings, get_model_path
from utils import setup_logging, cleanup_resources


@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
def cli(log_level):
    """Basketball Analytics CLI"""
    setup_logging(level=log_level)


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--start', default=0, help='Start frame')
@click.option('--end', default=None, type=int, help='End frame')
@click.option('--output', default='output.mp4', help='Output video path')
@click.option('--analytics', default='analytics.json', help='Analytics output path')
@click.option('--batch-size', default=20, help='Batch size for processing')
@click.option('--no-visualization', is_flag=True, help='Disable visualization')
@click.option('--enable-pose', is_flag=True, help='Enable pose estimation')
def analyze(video_path, start, end, output, analytics, batch_size, 
            no_visualization, enable_pose):
    """Analyze basketball video"""
    
    click.echo(f"Analyzing video: {video_path}")
    
    # Create config
    config = ProcessorConfig(
        yolo_model_path=get_model_path('yolo'),
        start_frame=start,
        end_frame=end,
        batch_size=batch_size,
        enable_visualization=not no_visualization,
        enable_pose_estimation=enable_pose,
        output_video_path=output if not no_visualization else None,
        analytics_output_path=analytics
    )
    
    # Process video
    processor = VideoProcessor(config)
    
    with click.progressbar(length=100) as bar:
        def progress_callback(p):
            bar.update(int(p * 100) - bar.pos)
            
        result = processor.process_video(video_path, progress_callback)
    
    # Summary
    click.echo(f"\nAnalysis complete!")
    click.echo(f"Frames processed: {result.processed_frames}")
    click.echo(f"Players tracked: {len(result.tracks)}")
    click.echo(f"Total possessions: {len(result.possessions)}")
    click.echo(f"Total events: {len(result.events)}")
    
    if config.analytics_output_path:
        click.echo(f"\nAnalytics saved to: {config.analytics_output_path}")
    
    if config.output_video_path and not no_visualization:
        click.echo(f"Output video saved to: {config.output_video_path}")
    
    cleanup_resources()


@cli.command()
@click.argument('analytics_path', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['summary', 'teams', 'possessions', 'events']), 
              default='summary', help='What to show')
def show(analytics_path, format):
    """Show analytics results"""
    
    with open(analytics_path, 'r') as f:
        data = json.load(f)
    
    if format == 'summary':
        click.echo("=== Analysis Summary ===")
        click.echo(f"Video: {data['video_info']['path']}")
        click.echo(f"FPS: {data['video_info']['fps']}")
        click.echo(f"Frames: {data['video_info']['processed_frames']}/{data['video_info']['total_frames']}")
        click.echo(f"\nTracking:")
        click.echo(f"  Total tracks: {data['tracking']['total_tracks']}")
        click.echo(f"  Unique players: {data['tracking']['unique_players']}")
        click.echo(f"\nAnalytics:")
        click.echo(f"  Possessions: {data['analytics']['total_possessions']}")
        click.echo(f"  Events: {data['analytics']['total_events']}")
        
    elif format == 'teams':
        click.echo("=== Team Statistics ===")
        team_stats = data.get('team_stats', {})
        for team_id, stats in team_stats.items():
            click.echo(f"\nTeam {team_id}:")
            for stat, value in stats.items():
                click.echo(f"  {stat}: {value}")
                
    elif format == 'possessions':
        click.echo("=== Possessions ===")
        # Implementation depends on data structure
        pass
        
    elif format == 'events':
        click.echo("=== Events ===")
        # Implementation depends on data structure
        pass


@cli.command()
def list_models():
    """List available models"""
    from config import ModelPaths
    model_paths = ModelPaths()
    models = model_paths.list_models()
    
    click.echo("=== Available Models ===")
    for model_type, versions in models.items():
        click.echo(f"\n{model_type}:")
        for version, info in versions.items():
            status = "✓" if info['exists'] else "✗"
            click.echo(f"  {status} {version}: {info['path']}")


if __name__ == '__main__':
    cli()
