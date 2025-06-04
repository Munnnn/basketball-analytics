"""
GPU and memory management utilities
"""

import gc
import torch
import psutil
from typing import Dict, Optional
import logging


def cleanup_resources():
    """Force cleanup of GPU and system resources"""
    # Python garbage collection
    gc.collect()
    
    # CUDA cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Clear OpenCV windows
    try:
        import cv2
        cv2.destroyAllWindows()
    except:
        pass


def get_gpu_memory() -> Optional[Dict[str, float]]:
    """Get GPU memory statistics"""
    if not torch.cuda.is_available():
        return None
    
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
        'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
        'total_mb': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
        'free_mb': (torch.cuda.get_device_properties(0).total_memory - 
                   torch.cuda.memory_allocated()) / 1024 / 1024
    }


def get_system_memory() -> Dict[str, float]:
    """Get system memory statistics"""
    memory = psutil.virtual_memory()
    
    return {
        'total_mb': memory.total / 1024 / 1024,
        'available_mb': memory.available / 1024 / 1024,
        'used_mb': memory.used / 1024 / 1024,
        'percent': memory.percent
    }


def get_available_memory() -> float:
    """Get available system memory in MB"""
    return psutil.virtual_memory().available / 1024 / 1024


def optimize_gpu_memory():
    """Optimize GPU memory settings"""
    if torch.cuda.is_available():
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except:
            pass
        
        device_name = torch.cuda.get_device_name(0)
        logging.info(f"GPU memory optimized: {device_name}")


class MemoryMonitor:
    """Monitor memory usage during processing"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = 0
        self.peak_memory = 0
        self.gpu_start = None
        self.gpu_peak = None
        
    def start_monitoring(self):
        """Start memory monitoring"""
        self.start_memory = self.get_memory_mb()
        self.peak_memory = self.start_memory
        
        if torch.cuda.is_available():
            self.gpu_start = get_gpu_memory()
            self.gpu_peak = self.gpu_start.copy() if self.gpu_start else None
            
    def get_memory_mb(self) -> float:
        """Get current process memory in MB"""
        return self.process.memory_info().rss / 1024 / 1024
        
    def check_memory(self, context: str = "") -> Dict[str, float]:
        """Check and update memory usage"""
        current_memory = self.get_memory_mb()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        result = {
            'current_mb': current_memory,
            'peak_mb': self.peak_memory,
            'increase_mb': current_memory - self.start_memory
        }
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_current = get_gpu_memory()
            if gpu_current and self.gpu_peak:
                self.gpu_peak['allocated_mb'] = max(
                    self.gpu_peak['allocated_mb'], 
                    gpu_current['allocated_mb']
                )
                result['gpu_current_mb'] = gpu_current['allocated_mb']
                result['gpu_peak_mb'] = self.gpu_peak['allocated_mb']
                
        if context:
            logging.info(f"Memory {context}: CPU={current_memory:.1f}MB")
            
        return result
        
    def get_summary(self) -> Dict[str, float]:
        """Get memory usage summary"""
        summary = {
            'cpu_start_mb': self.start_memory,
            'cpu_peak_mb': self.peak_memory,
            'cpu_increase_mb': self.peak_memory - self.start_memory
        }
        
        if self.gpu_start and self.gpu_peak:
            summary['gpu_start_mb'] = self.gpu_start['allocated_mb']
            summary['gpu_peak_mb'] = self.gpu_peak['allocated_mb']
            summary['gpu_increase_mb'] = (
                self.gpu_peak['allocated_mb'] - self.gpu_start['allocated_mb']
            )
            
        return summary
