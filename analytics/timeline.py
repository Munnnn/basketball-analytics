"""
Game timeline generation
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from core import PlayEvent, PossessionInfo, PlayClassification


class TimelineGenerator:
    """Generate game timeline with key events and statistics"""
    
    def __init__(self, fps: float = 30.0):
        """
        Initialize timeline generator
        
        Args:
            fps: Video frames per second
        """
        self.fps = fps
        self.timeline_events = []
        self.possession_changes = []
        self.play_classifications = []
        
    def add_event(self, event: PlayEvent):
        """Add event to timeline"""
        self.timeline_events.append({
            'type': 'event',
            'data': event,
            'timestamp': event.frame_idx / self.fps
        })
        
    def add_possession_change(self, possession: PossessionInfo):
        """Add possession change to timeline"""
        if possession.possession_change:
            self.possession_changes.append({
                'type': 'possession_change',
                'data': possession,
                'timestamp': possession.frame_idx / self.fps
            })
            
    def add_play_classification(self, play: PlayClassification):
        """Add play classification to timeline"""
        self.play_classifications.append({
            'type': 'play',
            'data': play,
            'timestamp': play.start_frame / self.fps,
            'duration': (play.end_frame - play.start_frame) / self.fps
        })
        
    def generate_timeline(self) -> Dict:
        """Generate complete timeline with statistics"""
        # Sort all events by timestamp
        all_events = (
            self.timeline_events +
            self.possession_changes +
            self.play_classifications
        )
        all_events.sort(key=lambda x: x['timestamp'])
        
        # Calculate team statistics
        team_stats = self._calculate_team_statistics()
        
        # Generate timeline markers
        markers = self._generate_timeline_markers()
        
        # Create timeline summary
        summary = {
            'total_events': len(all_events),
            'game_duration': self._get_game_duration(),
            'team_statistics': team_stats,
            'play_distribution': self._get_play_distribution(),
            'possession_summary': self._get_possession_summary()
        }
        
        return {
            'events': all_events,
            'markers': markers,
            'summary': summary
        }
        
    def _calculate_team_statistics(self) -> Dict:
        """Calculate statistics for each team"""
        stats = {
            0: defaultdict(int),
            1: defaultdict(int)
        }
        
        # Count events by team
        for event in self.timeline_events:
            if event['data'].team_id in [0, 1]:
                team_id = event['data'].team_id
                event_type = event['data'].type
                stats[team_id][event_type] += 1
                
        # Count possessions
        for possession in self.possession_changes:
            if possession['data'].team_id in [0, 1]:
                team_id = possession['data'].team_id
                stats[team_id]['possessions'] += 1
                
        # Count plays
        for play in self.play_classifications:
            if play['data'].team_id in [0, 1]:
                team_id = play['data'].team_id
                play_name = play['data'].play_name
                stats[team_id][f'play_{play_name}'] += 1
                
        return dict(stats)
        
    def _generate_timeline_markers(self) -> List[Dict]:
        """Generate visual timeline markers"""
        markers = []
        
        # Add possession markers
        current_possession = None
        possession_start = None
        
        for event in self.possession_changes:
            if current_possession is not None and possession_start is not None:
                # End previous possession
                markers.append({
                    'type': 'possession',
                    'team': current_possession,
                    'start_time': possession_start,
                    'end_time': event['timestamp'],
                    'duration': event['timestamp'] - possession_start
                })
                
            current_possession = event['data'].team_id
            possession_start = event['timestamp']
            
        # Add event markers
        for event in self.timeline_events:
            if event['data'].confidence > 0.7:
                markers.append({
                    'type': event['data'].type,
                    'team': event['data'].team_id,
                    'time': event['timestamp'],
                    'position': event['data'].position.tolist() if event['data'].position is not None else None
                })
                
        return markers
        
    def _get_game_duration(self) -> float:
        """Get total game duration in seconds"""
        if not self.timeline_events and not self.possession_changes:
            return 0.0
            
        all_timestamps = []
        
        for event in self.timeline_events:
            all_timestamps.append(event['timestamp'])
            
        for possession in self.possession_changes:
            all_timestamps.append(possession['timestamp'])
            
        for play in self.play_classifications:
            all_timestamps.append(play['timestamp'])
            all_timestamps.append(play['timestamp'] + play.get('duration', 0))
            
        return max(all_timestamps) - min(all_timestamps) if all_timestamps else 0.0
        
    def _get_play_distribution(self) -> Dict[str, int]:
        """Get distribution of play types"""
        distribution = defaultdict(int)
        
        for play in self.play_classifications:
            play_name = play['data'].play_name
            distribution[play_name] += 1
            
        return dict(distribution)
        
    def _get_possession_summary(self) -> Dict:
        """Get possession summary statistics"""
        if not self.possession_changes:
            return {
                'total_possessions': 0,
                'avg_possession_duration': 0.0
            }
            
        durations = []
        for i in range(1, len(self.possession_changes)):
            duration = (
                self.possession_changes[i]['timestamp'] -
                self.possession_changes[i-1]['timestamp']
            )
            durations.append(duration)
            
        return {
            'total_possessions': len(self.possession_changes),
            'avg_possession_duration': np.mean(durations) if durations else 0.0,
            'min_possession_duration': min(durations) if durations else 0.0,
            'max_possession_duration': max(durations) if durations else 0.0
        }
        
    def export_timeline(self, format: str = 'json') -> str:
        """Export timeline in specified format"""
        timeline_data = self.generate_timeline()
        
        if format == 'json':
            import json
            return json.dumps(timeline_data, indent=2, default=str)
        elif format == 'html':
            return self._generate_html_timeline(timeline_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def _generate_html_timeline(self, timeline_data: Dict) -> str:
        """Generate HTML visualization of timeline"""
        html = """
        <div class="timeline-container">
            <h2>Game Timeline</h2>
            <div class="timeline-stats">
                <p>Duration: {duration:.1f}s</p>
                <p>Total Events: {total_events}</p>
            </div>
            <div class="timeline-chart">
                <!-- Timeline visualization would go here -->
            </div>
        </div>
        """.format(
            duration=timeline_data['summary']['game_duration'],
            total_events=timeline_data['summary']['total_events']
        )
        
        return html
