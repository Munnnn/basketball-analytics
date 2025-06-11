"""
Data serialization utilities
"""

import json
import pickle
import numpy as np
from typing import Any, Dict
from pathlib import Path


class NumpyJsonEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)


class JsonSerializer:
    """JSON serialization utilities"""
    
    @staticmethod
    def save(data: Dict, filepath: str, indent: int = 2):
        """Save data to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, cls=NumpyJsonEncoder, indent=indent)
            
    @staticmethod
    def load(filepath: str) -> Dict:
        """Load data from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
            

class PickleSerializer:
    """Pickle serialization utilities"""
    
    @staticmethod
    def save(data: Any, filepath: str):
        """Save data to pickle file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
    @staticmethod
    def load(filepath: str) -> Any:
        """Load data from pickle file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
