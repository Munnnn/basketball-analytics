
"""
Batch processing optimization
"""

import numpy as np
from typing import List, Tuple, Optional, Generator, Dict
import logging
import psutil

from utils import get_available_memory


class BatchOptimizer:
    """Optimize batch processing for memory and performance"""

    def __init__(self,
                 target_memory_usage: float = 0.8,
                 min_batch_size: int = 1,
                 max_batch_size: int = 32):
        """
        Initialize batch optimizer

        Args:
            target_memory_usage: Target memory usage ratio (0-1)
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
        """
        self.target_memory_usage = target_memory_usage
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.processing_times = []
        self.batch_sizes = []

    def optimize_batch_size(self,
                          frame_size: Tuple[int, int, int],
                          model_memory_mb: float = 500) -> int:
        """
        Calculate optimal batch size based on available resources

        Args:
            frame_size: Frame dimensions (height, width, channels)
            model_memory_mb: Estimated model memory usage in MB

        Returns:
            Optimal batch size
        """
        # Calculate frame memory
        frame_memory_mb = np.prod(frame_size) * 4 / (1024 * 1024)  # Float32

        # Get available memory
        available_memory_mb = get_available_memory()

        # Calculate usable memory
        usable_memory_mb = available_memory_mb * self.target_memory_usage

        # Reserve memory for model and overhead
        processing_memory_mb = usable_memory_mb - model_memory_mb - 500  # 500MB overhead

        if processing_memory_mb <= 0:
            self.logger.warning("Insufficient memory, using minimum batch size")
            return self.min_batch_size

        # Calculate batch size
        # Account for intermediate results (detection, masks, etc.)
        memory_per_frame = frame_memory_mb * 3  # Conservative estimate
        optimal_batch = int(processing_memory_mb / memory_per_frame)

        # Apply constraints
        optimal_batch = max(self.min_batch_size, min(optimal_batch, self.max_batch_size))

        self.logger.info(f"Optimal batch size: {optimal_batch} "
                        f"(available: {available_memory_mb:.1f}MB, "
                        f"frame: {frame_memory_mb:.1f}MB)")

        return optimal_batch

    def adaptive_batch_generator(self,
                               items: List,
                               initial_batch_size: Optional[int] = None) -> Generator:
        """
        Generate batches with adaptive sizing based on performance

        Args:
            items: List of items to batch
            initial_batch_size: Initial batch size (None for auto)

        Yields:
            Batches of items
        """
        if initial_batch_size is None:
            current_batch_size = self.min_batch_size
        else:
            current_batch_size = initial_batch_size

        i = 0
        while i < len(items):
            # Get current batch
            batch = items[i:i + current_batch_size]

            # Yield batch and get processing time
            start_time = time.time()
            yield batch
            processing_time = time.time() - start_time

            # Track performance
            self.processing_times.append(processing_time)
            self.batch_sizes.append(len(batch))

            # Adapt batch size
            if len(self.processing_times) >= 3:
                current_batch_size = self._adapt_batch_size(current_batch_size)

            i += len(batch)

    def _adapt_batch_size(self, current_size: int) -> int:
        """Adapt batch size based on recent performance"""
        # Calculate recent performance metrics
        recent_times = self.processing_times[-5:]
        recent_sizes = self.batch_sizes[-5:]

        # Calculate processing rate (items/second)
        rates = [size / time for size, time in zip(recent_sizes, recent_times)]
        avg_rate = np.mean(rates)

        # Check memory usage
        memory_usage = psutil.virtual_memory().percent / 100

        # Adapt size
        if memory_usage > 0.9:  # High memory usage
            new_size = int(current_size * 0.8)
        elif memory_usage < 0.6 and avg_rate > 10:  # Low memory, good performance
            new_size = int(current_size * 1.2)
        else:
            new_size = current_size

        # Apply constraints
        new_size = max(self.min_batch_size, min(new_size, self.max_batch_size))

        if new_size != current_size:
            self.logger.info(f"Adapting batch size: {current_size} -> {new_size}")

        return new_size

    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if not self.processing_times:
            return {}

        return {
            'total_batches': len(self.processing_times),
            'avg_batch_size': np.mean(self.batch_sizes),
            'avg_processing_time': np.mean(self.processing_times),
            'total_processing_time': sum(self.processing_times),
            'avg_throughput': sum(self.batch_sizes) / sum(self.processing_times)
        }
