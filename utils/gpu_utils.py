"""
GPU, system memory management, and resource cleanup utilities.

This is the canonical module for all memory/resource operations.
"""

import gc
import logging
from typing import Dict, Optional

import psutil
import torch


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resource cleanup
# ---------------------------------------------------------------------------

def cleanup_resources():
    """Force cleanup of GPU and system resources."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        import cv2
        cv2.destroyAllWindows()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Memory info helpers
# ---------------------------------------------------------------------------

def get_memory_info() -> Dict[str, float]:
    """Get current process and GPU memory usage."""
    process = psutil.Process()
    mem = process.memory_info()
    info: Dict[str, float] = {
        'rss_mb': mem.rss / 1024 / 1024,
        'vms_mb': mem.vms / 1024 / 1024,
        'percent': process.memory_percent(),
    }
    if torch.cuda.is_available():
        info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
    return info


def get_gpu_memory() -> Optional[Dict[str, float]]:
    """Get GPU memory statistics (None if no GPU)."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
        'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
        'total_mb': props.total_memory / 1024 / 1024,
        'free_mb': (props.total_memory - torch.cuda.memory_allocated()) / 1024 / 1024,
    }


def get_system_memory() -> Dict[str, float]:
    """Get system memory statistics."""
    memory = psutil.virtual_memory()
    return {
        'total_mb': memory.total / 1024 / 1024,
        'available_mb': memory.available / 1024 / 1024,
        'used_mb': memory.used / 1024 / 1024,
        'percent': memory.percent,
    }


def get_available_memory() -> float:
    """Get available system memory in MB."""
    return psutil.virtual_memory().available / 1024 / 1024


# ---------------------------------------------------------------------------
# Cleanup / optimization
# ---------------------------------------------------------------------------

def cleanup_memory(log_step: Optional[str] = None) -> Dict[str, float]:
    """Force memory cleanup with optional before/after logging."""
    before = get_memory_info()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    after = get_memory_info()
    if log_step:
        logger.info(
            f"Memory cleanup at {log_step}: "
            f"{before['rss_mb']:.1f}MB -> {after['rss_mb']:.1f}MB "
            f"(freed {before['rss_mb'] - after['rss_mb']:.1f}MB)"
        )
    return after


def check_memory_threshold(threshold_percent: float = 85.0) -> bool:
    """Return True if memory usage exceeds *threshold_percent*."""
    info = get_memory_info()
    if info['percent'] > threshold_percent:
        logger.warning(f"High memory usage: {info['percent']:.1f}% ({info['rss_mb']:.1f}MB)")
        return True
    return False


def optimize_gpu_memory():
    """Set GPU memory flags for efficient usage."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.set_per_process_memory_fraction(0.8)
    except RuntimeError:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except AttributeError:
        pass
    logger.info(f"GPU memory optimized: {torch.cuda.get_device_name(0)}")


def force_cleanup_large_objects(*objects):
    """Delete specific objects and run cleanup."""
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def optimize_memory_usage():
    """Comprehensive memory optimization (GC + GPU)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Context managers / monitors
# ---------------------------------------------------------------------------

class MemoryMonitor:
    """Monitor memory usage during processing."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_memory: float = 0
        self.peak_memory: float = 0
        self.gpu_start: Optional[Dict[str, float]] = None
        self.gpu_peak: Optional[Dict[str, float]] = None

    def start_monitoring(self):
        self.start_memory = self.get_memory_mb()
        self.peak_memory = self.start_memory
        if torch.cuda.is_available():
            self.gpu_start = get_gpu_memory()
            self.gpu_peak = self.gpu_start.copy() if self.gpu_start else None

    def get_memory_mb(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024

    def check_memory(self, context: str = "") -> Dict[str, float]:
        current = self.get_memory_mb()
        self.peak_memory = max(self.peak_memory, current)
        result: Dict[str, float] = {
            'current_mb': current,
            'peak_mb': self.peak_memory,
            'increase_mb': current - self.start_memory,
        }
        if torch.cuda.is_available():
            gpu_current = get_gpu_memory()
            if gpu_current and self.gpu_peak:
                self.gpu_peak['allocated_mb'] = max(
                    self.gpu_peak['allocated_mb'], gpu_current['allocated_mb']
                )
                result['gpu_current_mb'] = gpu_current['allocated_mb']
                result['gpu_peak_mb'] = self.gpu_peak['allocated_mb']
        if context:
            logger.info(f"Memory {context}: CPU={current:.1f}MB")
        return result

    def get_summary(self) -> Dict[str, float]:
        summary: Dict[str, float] = {
            'cpu_start_mb': self.start_memory,
            'cpu_peak_mb': self.peak_memory,
            'cpu_increase_mb': self.peak_memory - self.start_memory,
        }
        if self.gpu_start and self.gpu_peak:
            summary['gpu_start_mb'] = self.gpu_start['allocated_mb']
            summary['gpu_peak_mb'] = self.gpu_peak['allocated_mb']
            summary['gpu_increase_mb'] = (
                self.gpu_peak['allocated_mb'] - self.gpu_start['allocated_mb']
            )
        return summary


class MemoryManager:
    """Context manager for memory-intensive operations."""

    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_memory: Optional[Dict[str, float]] = None

    def __enter__(self):
        self.start_memory = get_memory_info()
        logger.debug(f"Starting {self.operation_name}: {self.start_memory['rss_mb']:.1f}MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_memory(self.operation_name)
        end = get_memory_info()
        logger.debug(f"Finished {self.operation_name}: {end['rss_mb']:.1f}MB")
