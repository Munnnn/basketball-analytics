"""
JSON serialization utilities
"""

import json
import numpy as np
from collections import defaultdict
from typing import Any


def safe_json_convert(obj: Any) -> Any:
    """Convert object to JSON-serializable format"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, defaultdict):
        return dict(obj)
    elif isinstance(obj, dict):
        return {str(key): safe_json_convert(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_convert(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return safe_json_convert(obj.__dict__)
    elif obj is None or isinstance(obj, (bool, str, int, float)):
        return obj
    else:
        return str(obj)


def create_json_serializable(data: Any) -> Any:
    """Create JSON-serializable version of data"""
    return json.loads(json.dumps(data, default=safe_json_convert))


def pretty_json(data: Any, indent: int = 2) -> str:
    """Convert data to pretty-printed JSON string"""
    return json.dumps(
        data, 
        default=safe_json_convert,
        indent=indent,
        sort_keys=True
    )


def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, default=safe_json_convert, indent=indent)


def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)
