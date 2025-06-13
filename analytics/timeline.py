"""
Game timeline generation - Basketball Enhanced
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from core import PlayEvent, PossessionInfo, PlayClassification


class TimelineGenerator:
    """Generate game timeline with key events and statistics - Basketball Enhanced"""

    def __init__(self, fps: float = 30.0, basketball_timeline: bool = True):
        """
        Initialize timeline generator

        Args:
            fps: Video frames per second
            basketball_timeline: Enable basketball-specific timeline features
        """
        self.fps = fps
        self.basketball_timeline = basketball_timeline
        self.timeline_events = []
        self.possession_changes = []
        self.play_classifications = []

        # Basketball-specific tracking
        self.basketball_stats = {
            'team_possessions': {0: 0, 1: 0},
            'basketball_events': defaultdict(int),
            'quarter_breaks': [],
            'key_basketball_moments': []
        }

    def add_event(self, event: PlayEvent):
        """Add event to timeline"""
        self.timeline_events.append({
            'type': 'event',
            'data': event,
            'timestamp': event.frame_idx / self.fps
        })

        # Basketball-specific event tracking
        if self.basketball_timeline:
            self._track_basketball_event(event)

    def add_possession_change(self, possession: PossessionInfo):
        """Add possession change to timeline"""
        if possession.possession_change:
            self.possession_changes.append({
                'type': 'possession_change',
                'data': possession,
                'timestamp': possession.frame_idx / self.fps
            })

            # Basketball possession tracking
            if self.basketball_timeline:
                self._track_basketball_possession(possession)

    def add_play_classification(self, play: PlayClassification):
        """Add play classification to timeline"""
        self.play_classifications.append({
            'type': 'play',
            'data': play,
            'timestamp': play.start_frame / self.fps,
            'duration': (play.end_frame - play.start_frame) / self.fps
        })

        # Basketball play tracking
        if self.basketball_timeline:
            self._track_basketball_play(play)

    def generate_basketball_timeline(self) -> Dict:
        """Generate basketball-enhanced timeline - THE MISSING METHOD"""
        if not self.basketball_timeline:
            # Fallback to regular timeline
            return self.generate_timeline()

        # Generate basketball-specific timeline
        timeline_data = self.generate_timeline()

        # Add basketball enhancements
        timeline_data['basketball_stats'] = self.basketball_stats.copy()
        timeline_data['basketball_enhanced'] = True

        # Add basketball-specific summary
        timeline_data['summary']['basketball_analysis'] = self._generate_basketball_summary()

        # Add basketball timeline markers
        timeline_data['basketball_markers'] = self._generate_basketball_markers()

        return timeline_data

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
            'possession_summary': self._get_possession_summary(),
            'possessions': self._get_all_possessions(),
            'plays': self._get_all_plays()
        }

        return {
            'events': all_events,
            'markers': markers,
            'summary': summary
        }

    def _track_basketball_event(self, event: PlayEvent):
        """Track basketball-specific events"""
        event_type = event.type

        # Count basketball events
        self.basketball_stats['basketball_events'][event_type] += 1

        # Track key moments
        if event.confidence > 0.8:
            self.basketball_stats['key_basketball_moments'].append({
                'type': event_type,
                'timestamp': event.frame_idx / self.fps,
                'team': event.team_id,
                'confidence': event.confidence
            })

    def _track_basketball_possession(self, possession: PossessionInfo):
        """Track basketball possession changes"""
        if possession.team_id in [0, 1]:
            self.basketball_stats['team_possessions'][possession.team_id] += 1

    def _track_basketball_play(self, play: PlayClassification):
        """Track basketball play classifications"""
        # This can be extended with basketball-specific play tracking
        pass

    def _generate_basketball_summary(self) -> Dict:
        """Generate basketball-specific summary"""
        total_possessions = sum(self.basketball_stats['team_possessions'].values())

        return {
            'total_basketball_possessions': total_possessions,
            'team_possession_balance': self.basketball_stats['team_possessions'],
            'basketball_events_count': dict(self.basketball_stats['basketball_events']),
            'key_moments_count': len(self.basketball_stats['key_basketball_moments']),
            'possession_percentage': {
                team_id: (count / total_possessions * 100) if total_possessions > 0 else 0
                for team_id, count in self.basketball_stats['team_possessions'].items()
            }
        }

    def _generate_basketball_markers(self) -> List[Dict]:
        """Generate basketball-specific timeline markers"""
        markers = []

        # Add key basketball moments as markers
        for moment in self.basketball_stats['key_basketball_moments']:
            markers.append({
                'type': 'basketball_key_moment',
                'event_type': moment['type'],
                'team': moment['team'],
                'time': moment['timestamp'],
                'confidence': moment['confidence'],
                'basketball_specific': True
            })

        return markers

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

    def _get_all_possessions(self) -> List[Dict]:
        """Get all possession data for summary"""
        possessions = []
        for possession in self.possession_changes:
            possessions.append({
                'team_id': possession['data'].team_id,
                'timestamp': possession['timestamp'],
                'confidence': possession['data'].confidence
            })
        return possessions

    def _get_all_plays(self) -> List[Dict]:
        """Get all play data for summary"""
        plays = []
        for play in self.play_classifications:
            plays.append({
                'play_name': play['data'].play_name,
                'team_id': play['data'].team_id,
                'start_time': play['timestamp'],
                'duration': play.get('duration', 0),
                'confidence': play['data'].confidence
            })
        return plays

    def export_timeline(self, format: str = 'json') -> str:
        """Export timeline in specified format"""
        if self.basketball_timeline:
            timeline_data = self.generate_basketball_timeline()
        else:
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
        basketball_info = ""
        if self.basketball_timeline and 'basketball_stats' in timeline_data:
            basketball_info = f"""
            <div class="basketball-stats">
                <h3>Basketball Statistics</h3>
                <p>Team Possessions: {timeline_data['basketball_stats']['team_possessions']}</p>
                <p>Key Moments: {len(timeline_data['basketball_stats']['key_basketball_moments'])}</p>
            </div>
            """

        html = f"""
        <div class="timeline-container">
            <h2>Game Timeline {'(Basketball Enhanced)' if self.basketball_timeline else ''}</h2>
            <div class="timeline-stats">
                <p>Duration: {timeline_data['summary']['game_duration']:.1f}s</p>
                <p>Total Events: {timeline_data['summary']['total_events']}</p>
            </div>
            {basketball_info}
            <div class="timeline-chart">
                <!-- Timeline visualization would go here -->
            </div>
        </div>
        """

        return html
