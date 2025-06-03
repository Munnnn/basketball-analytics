"""
Streaming data handling for large videos
"""

import pickle
import json
import os
from typing import Any, Optional, Generator
import logging


class StreamingWriter:
    """Write data to file in streaming fashion"""
    
    def __init__(self, filepath: str, format: str = 'pickle'):
        """
        Initialize streaming writer
        
        Args:
            filepath: Output file path
            format: Output format ('pickle' or 'jsonl')
        """
        self.filepath = filepath
        self.format = format
        self.file_handle = None
        self.items_written = 0
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
    def __enter__(self):
        mode = 'wb' if self.format == 'pickle' else 'w'
        self.file_handle = open(self.filepath, mode)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
        logging.info(f"Streaming writer closed: {self.items_written} items written")
        
    def write(self, item: Any):
        """Write single item to file"""
        if not self.file_handle:
            raise RuntimeError("Writer not initialized. Use with context manager.")
            
        if self.format == 'pickle':
            pickle.dump(item, self.file_handle)
        elif self.format == 'jsonl':
            json.dump(item, self.file_handle)
            self.file_handle.write('\n')
        else:
            raise ValueError(f"Unsupported format: {self.format}")
            
        self.items_written += 1
        
        # Periodic flush
        if self.items_written % 100 == 0:
            self.file_handle.flush()


class StreamingReader:
    """Read data from file in streaming fashion"""
    
    def __init__(self, filepath: str, format: str = 'pickle'):
        """
        Initialize streaming reader
        
        Args:
            filepath: Input file path
            format: Input format ('pickle' or 'jsonl')
        """
        self.filepath = filepath
        self.format = format
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
    def read_all(self) -> Generator[Any, None, None]:
        """Read all items from file"""
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
            else:
                raise ValueError(f"Unsupported format: {self.format}")
