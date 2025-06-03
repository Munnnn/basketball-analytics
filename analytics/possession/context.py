"""
Possession context tracking for enhanced analysis
"""

import numpy as np
from typing import Dict, List, Deque
from collections import deque, defaultdict
import time


class PossessionContext:
    """Track possession context including momentum and patterns"""
    
    def __init__(self, context_window: int = 3):
        """
        Initialize possession context tracker
        
        Args:
            context_window: Number of recent plays to track
        """
        self.context_window = context_window
        self.recent_plays = deque(maxlen=context_window)
        self.score_tracker = {0: 0, 1: 0}
        self.momentum_tracker = {0: 0.0, 1: 0.0}
        self.game_state = {
            'quarter': 1,
            'time_remaining': 720,
            'last_score_team': None,
            'last_score_time': None,
            'fast_break_streak': {0: 0, 1: 0}
        }
        
    def add_completed_play(self, play_data: Dict):
        """Add completed play to context"""
        self.recent_plays.append({
            'play_type': play_data.get('play_type'),
            'team_id': play_data.get('team_id'),
            'duration': play_data.get('duration'),
            'outcome': play_data.get('outcome', 'unknown'),
            'timestamp': time.time(),
            'frame': play_data.get('frame', 0)
        })
        
        self._update_momentum(play_data)
        
    def _update_momentum(self, play_data: Dict):
        """Update team momentum based on play outcome"""
        team_id = play_data.get('team_id')
        if team_id not in [0, 1]:
            return
            
        outcome = play_data.get('outcome', 'unknown')
        play_type = play_data.get('play_type')
        
        momentum_change = 0.0
        
        # Calculate momentum change
        if outcome == 'score':
            momentum_change = 0.3
            if play_type == 3:  # Fast break
                momentum_change = 0.5
                self.game_state['fast_break_streak'][team_id] += 1
            else:
                self.game_state['fast_break_streak'][team_id] = 0
                
        elif outcome == 'turnover':
            momentum_change = -0.4
            self.game_state['fast_break_streak'][team_id] = 0
            
        elif outcome == 'defensive_stop':
            momentum_change = 0.2
            
        # Apply momentum change with decay
        self.momentum_tracker[team_id] = np.clip(
            self.momentum_tracker[team_id] * 0.8 + momentum_change,
            -1.0, 1.0
        )
        
        # Opposite team gets inverse momentum
        other_team = 1 - team_id
        self.momentum_tracker[other_team] = np.clip(
            self.momentum_tracker[other_team] * 0.9 - momentum_change * 0.5,
            -1.0, 1.0
        )
        
    def get_context(self) -> Dict:
        """Get current possession context"""
        recent_play_types = [play['play_type'] for play in self.recent_plays]
        recent_teams = [play['team_id'] for play in self.recent_plays]
        
        return {
            'recent_play_types': recent_play_types,
            'recent_teams': recent_teams,
            'team_momentum': dict(self.momentum_tracker),
            'score_differential': self.score_tracker[0] - self.score_tracker[1],
            'fast_break_streak': dict(self.game_state['fast_break_streak']),
            'possession_frequency': self._calculate_possession_frequency(),
            'tempo': self._calculate_tempo()
        }
        
    def _calculate_possession_frequency(self) -> Dict[int, float]:
        """Calculate possession frequency for each team"""
        if not self.recent_plays:
            return {0: 0.5, 1: 0.5}
            
        team_counts = defaultdict(int)
        for play in self.recent_plays:
            team_id = play.get('team_id')
            if team_id in [0, 1]:
                team_counts[team_id] += 1
                
        total = sum(team_counts.values())
        if total == 0:
            return {0: 0.5, 1: 0.5}
            
        return {
            0: team_counts[0] / total,
            1: team_counts[1] / total
        }
        
    def _calculate_tempo(self) -> float:
        """Calculate game tempo based on possession durations"""
        if len(self.recent_plays) < 2:
            return 1.0
            
        durations = [play['duration'] for play in self.recent_plays if play['duration'] > 0]
        if not durations:
            return 1.0
            
        avg_duration = np.mean(durations)
        
        # Fast tempo: < 8 seconds, Slow tempo: > 20 seconds
        if avg_duration < 8:
            return 1.5  # Fast
        elif avg_duration > 20:
            return 0.5  # Slow
        else:
            return 1.0  # Normal
            
    def predict_next_play(self, current_team: int) -> Dict[str, float]:
        """Predict likely next play type based on context"""
        context = self.get_context()
        momentum = context['team_momentum'].get(current_team, 0.0)
        
        predictions = {
            'fast_break': 0.1,
            'transition': 0.2,
            'half_court': 0.6,
            'post_up': 0.1
        }
        
        # Adjust based on momentum
        if momentum > 0.3:
            predictions['fast_break'] = 0.3
            predictions['transition'] = 0.4
            predictions['half_court'] = 0.3
        elif momentum < -0.3:
            predictions['half_court'] = 0.7
            predictions['post_up'] = 0.2
            predictions['transition'] = 0.1
            
        return predictions
