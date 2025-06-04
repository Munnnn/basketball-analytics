"""
Streaming data handlers
"""

import json
import pickle
from typing import Any, Generator
from pathlib import Path
import logging

from .serialization import NumpyJsonEncoder


class StreamingWriter:
    """Write data in streaming fashion"""
    
    def __init__(self, filepath: str, format: str = 'pickle'):
        """
        Initialize streaming writer
        
        Args:
            filepath: Output file path
            format: 'pickle' or 'jsonl'
        """
        self.filepath = filepath
        self.format = format
        self.file_handle = None
        self.items_written = 0
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def __enter__(self):
        mode = 'wb' if self.format == 'pickle' else 'w'
        self.file_handle = open(self.filepath, mode)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def write(self, item: Any):
        """Write single item"""
        if not self.file_handle:
            raise RuntimeError("Writer not opened")
            
        if self.format == 'pickle':
            pickle.dump(item, self.file_handle)
        elif self.format == 'jsonl':
            json.dump(item, self.file_handle, cls=NumpyJsonEncoder)
            self.file_handle.write('\n')
            
        self.items_written += 1
        
        # Periodic flush
        if self.items_written % 100 == 0:
            self.file_handle.flush()
            
    def close(self):
        """Close writer"""
        if self.file_handle:
            self.file_handle.close()
            self.logger.info(f"Streaming writer closed: {self.items_written} items")
            

class StreamingReader:
    """Read data in streaming fashion"""
    
    def __init__(self, filepath: str, format: str = 'pickle'):
        """
        Initialize streaming reader
        
        Args:
            filepath: Input file path
            format: 'pickle' or 'jsonl'
        """
        self.filepath = filepath
        self.format = format
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
    def read(self) -> Generator[Any, None, None]:
        """Read items generator"""
        mode = 'rb' if self.format == 'pickle' else 'r'
        
        with open(self.filepath, mode) as f:
            if self.format == 'pickle':
                while True:
                    try:
                        yield pickle.load(f)
                    except EOFError:
                        break
            elif self.format == 'jsonl':
                for line in f:
                    if line.strip():
                        yield json.loads(line)
