"""
Color management for visualization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import colorsys


class ColorPalette:
    """Manage color palettes for visualization"""
    
    def __init__(self, colors: List[Tuple[int, int, int]]):
        """
        Initialize color palette
        
        Args:
            colors: List of BGR colors
        """
        self.colors = colors
        
    @classmethod
    def from_hex(cls, hex_colors: List[str]) -> 'ColorPalette':
        """Create palette from hex colors"""
        colors = []
        for hex_color in hex_colors:
            # Remove # if present
            hex_color = hex_color.lstrip('#')
            # Convert to RGB then BGR
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            colors.append((b, g, r))  # BGR for OpenCV
        return cls(colors)
        
    def get_color(self, index: int) -> Tuple[int, int, int]:
        """Get color by index with wraparound"""
        return self.colors[index % len(self.colors)]
        
    @classmethod
    def generate_distinct_colors(cls, n_colors: int) -> 'ColorPalette':
        """Generate distinct colors using HSV space"""
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            # Use high saturation and value for vibrant colors
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return cls(colors)


class TeamColorManager:
    """Manage team-specific color schemes"""
    
    def __init__(self):
        """Initialize team color manager"""
        # Define team color palettes
        self.team_palettes = {
            0: ColorPalette.from_hex([  # Team 0 - Red/Warm colors
                '#FF0000', '#DC143C', '#B22222', '#CD5C5C', '#F08080',
                '#FF4500', '#FF6347', '#FF7F50', '#FFA500', '#FFB347'
            ]),
            1: ColorPalette.from_hex([  # Team 1 - Blue/Cool colors
                '#0000FF', '#4169E1', '#1E90FF', '#6495ED', '#87CEEB',
                '#00BFFF', '#00CED1', '#20B2AA', '#48D1CC', '#40E0D0'
            ]),
            'default': ColorPalette.from_hex([  # Default/Mixed colors
                '#9370DB', '#808080', '#E6E6FA', '#FFD700', '#32CD32',
                '#DA70D6', '#98FB98', '#DDA0DD', '#F0E68C', '#D3D3D3'
            ]),
            'referee': ColorPalette.from_hex([  # Referee colors
                '#FFFF00', '#FFD700', '#FFA500', '#FF8C00', '#FF7F50'
            ])
        }
        
        # Special colors
        self.special_colors = {
            'ball': (0, 255, 255),      # Yellow
            'hoop': (0, 165, 255),      # Orange
            'backboard': (128, 128, 128) # Gray
        }
        
    def get_team_palette(self, team_id: Optional[int]) -> ColorPalette:
        """Get color palette for team"""
        if team_id in self.team_palettes:
            return self.team_palettes[team_id]
        return self.team_palettes['default']
        
    def get_track_color(self, track: 'Track') -> Tuple[int, int, int]:
        """Get color for specific track"""
        palette = self.get_team_palette(track.team_id)
        return palette.get_color(track.id)
        
    def get_detection_color(self, detection: 'Detection', 
                          track_id: Optional[int] = None,
                          team_id: Optional[int] = None) -> Tuple[int, int, int]:
        """Get color for detection"""
        # Special colors for specific classes
        if detection.class_id == 1:  # Ball
            return self.special_colors['ball']
        elif detection.class_id == 3:  # Hoop
            return self.special_colors['hoop']
        elif detection.class_id == 0:  # Backboard
            return self.special_colors['backboard']
        elif detection.class_id == 6:  # Referee
            palette = self.team_palettes['referee']
            return palette.get_color(0)
            
        # Team colors for players
        if team_id is not None:
            palette = self.get_team_palette(team_id)
            index = track_id if track_id is not None else 0
            return palette.get_color(index)
            
        # Default color
        return self.team_palettes['default'].get_color(0)
