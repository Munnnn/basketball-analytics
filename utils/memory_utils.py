"""
Backward-compatibility shim — all memory utilities now live in gpu_utils.py.
"""

from .gpu_utils import (  # noqa: F401
    get_memory_info,
    cleanup_memory,
    check_memory_threshold,
    optimize_memory_usage,
    force_cleanup_large_objects,
    MemoryManager,
)
