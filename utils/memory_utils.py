"""Memory optimization utilities for basketball analytics"""

import gc
import torch
import psutil
import logging

def get_memory_info():
    """Get current memory usage information"""
    process = psutil.Process()
    mem_info = process.memory_info()
    
    info = {
        'rss_mb': mem_info.rss / 1024 / 1024,
        'vms_mb': mem_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }
    
    if torch.cuda.is_available():
        info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
    
    return info

def cleanup_memory(log_step=None):
    """Force memory cleanup"""
    before = get_memory_info()
    
    # Python garbage collection
    gc.collect()
    
    # CUDA cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    after = get_memory_info()
    
    if log_step:
        logging.info(f"Memory cleanup at {log_step}: "
                    f"{before['rss_mb']:.1f}MB -> {after['rss_mb']:.1f}MB "
                    f"(freed {before['rss_mb'] - after['rss_mb']:.1f}MB)")
    
    return after

def check_memory_threshold(threshold_percent=85):
    """Check if memory usage exceeds threshold"""
    mem_info = get_memory_info()
    
    if mem_info['percent'] > threshold_percent:
        logging.warning(f"High memory usage: {mem_info['percent']:.1f}% "
                       f"({mem_info['rss_mb']:.1f}MB)")
        return True
    
    return False
