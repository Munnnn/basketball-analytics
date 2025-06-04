"""
Basic annotation components
"""

from .shapes import ShapeAnnotator
from .labels import LabelAnnotator
from .masks import MaskAnnotator

__all__ = ['ShapeAnnotator', 'LabelAnnotator', 'MaskAnnotator']
