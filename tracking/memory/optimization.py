"""
Memory optimization utilities
"""

import gc
import torch
import psutil
from typing import Optional
import logging


class MemoryMonitor:
    """Monitor memory usage during processing"""
    
    def __init__(self):
        """Initialize memory monitor"""
        self.process = psutil.Process()
        self.peak_memory = 0
        self.start_memory = 0
        
    def start_monitoring(self):
        """Start memory monitoring"""
        self.start_memory = self.get_memory_mb()
        self.peak_memory = self.start_memory
        logging.info(f"Memory monitoring started: {self.start_memory:.1f} MB")
        
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
        
    def check_memory(self, context: str = "") -> float:
        """Check and log current memory usage"""
        current_memory = self.get_memory_mb()
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
            
        if context:
            logging.info(f"Memory {context}: {current_memory:.1f} MB (Peak: {self.peak_memory:.1f} MB)")
            
        return current_memory
        
    def get_summary(self) -> dict:
        """Get memory usage summary"""
        return {
            'start_memory_mb': self.start_memory,
            'peak_memory_mb': self.peak_memory,
            'current_memory_mb': self.get_memory_mb(),
            'memory_increase_mb': self.peak_memory - self.start_memory
        }


class MemoryOptimizer:
    """Optimize memory usage during processing"""
    
    def __init__(self, cleanup_interval: int = 50):
        """
        Initialize memory optimizer
        
        Args:
            cleanup_interval: Frames between cleanups
        """
        self.cleanup_interval = cleanup_interval
        self.cleanup_counter = 0
        
    def cleanup(self, force: bool = False):
        """
        Perform memory cleanup
        
        Args:
            force: Force cleanup regardless of interval
        """
        self.cleanup_counter += 1
        
        if force or self.cleanup_counter >= self.cleanup_interval:
            # Python garbage collection
            gc.collect()
            
            # CUDA memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            self.cleanup_counter = 0
            logging.debug("Memory cleanup performed")
            
    def optimize_batch_size(self, available_memory_mb: float, 
                          frame_size_mb: float,
                          min_batch: int = 1,
                          max_batch: int = 32) -> int:
        """
        Calculate optimal batch size based on available memory
        
        Args:
            available_memory_mb: Available memory in MB
            frame_size_mb: Size of single frame in MB
            min_batch: Minimum batch size
            max_batch: Maximum batch size
            
        Returns:
            Optimal batch size
        """
        # Reserve 20% for overhead
        usable_memory = available_memory_mb * 0.8
        
        # Calculate batch size
        optimal_batch = int(usable_memory / frame_size_mb)
        
        # Apply constraints
        optimal_batch = max(min_batch, min(optimal_batch, max_batch))
        
        return optimal_batch
        
    @staticmethod
    def get_available_memory() -> float:
        """Get available system memory in MB"""
        memory = psutil.virtual_memory()
        return memory.available / 1024 / 1024
        
    @staticmethod
    def get_gpu_memory() -> Optional[dict]:
        """Get GPU memory info if available"""
        if not torch.cuda.is_available():
            return None
            
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            'total_mb': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        }
