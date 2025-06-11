"""
Batch processing optimization with basketball-specific enhancements
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Generator, Dict
import logging
import psutil

from utils import get_available_memory


class BatchOptimizer:
    """Optimize batch processing for basketball video analysis"""

    def __init__(self,
                 target_memory_usage: float = 0.8,
                 min_batch_size: int = 1,
                 max_batch_size: int = 32,
                 basketball_optimized: bool = True):
        """
        Initialize batch optimizer with basketball enhancements

        Args:
            target_memory_usage: Target memory usage ratio (0-1)
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            basketball_optimized: Enable basketball-specific optimizations
        """
        self.target_memory_usage = target_memory_usage
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.basketball_optimized = basketball_optimized

        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.processing_times = []
        self.batch_sizes = []
        
        # Basketball-specific tracking
        self.basketball_stats = {
            'total_players_processed': 0,
            'team_classification_batches': 0,
            'pose_estimation_batches': 0,
            'memory_optimizations_applied': 0
        }

    def optimize_basketball_batch_size(self,
                                     frame_size: Tuple[int, int, int],
                                     model_memory_mb: float = 500,
                                     enable_pose_estimation: bool = False,
                                     enable_team_classification: bool = True) -> int:
        """
        Calculate optimal batch size for basketball analysis

        Args:
            frame_size: Frame dimensions (height, width, channels)
            model_memory_mb: Estimated model memory usage in MB
            enable_pose_estimation: Whether pose estimation is enabled
            enable_team_classification: Whether team classification is enabled

        Returns:
            Optimal batch size for basketball processing
        """
        # Calculate frame memory
        frame_memory_mb = np.prod(frame_size) * 4 / (1024 * 1024)  # Float32

        # Get available memory
        available_memory_mb = get_available_memory()

        # Calculate usable memory
        usable_memory_mb = available_memory_mb * self.target_memory_usage

        # Basketball-specific memory adjustments
        basketball_overhead = 200  # Base basketball processing overhead
        
        if enable_team_classification:
            basketball_overhead += 300  # Team classification (ML models, crops)
            
        if enable_pose_estimation:
            basketball_overhead += 400  # Pose estimation memory
            
        # Reserve memory for model and basketball overhead
        processing_memory_mb = usable_memory_mb - model_memory_mb - basketball_overhead

        if processing_memory_mb <= 0:
            self.logger.warning("Insufficient memory for basketball processing, using minimum batch size")
            return self.min_batch_size

        # Calculate batch size for basketball processing
        # Account for intermediate results (detection, masks, crops, poses, etc.)
        memory_per_frame = frame_memory_mb * 4  # More conservative for basketball
        
        if enable_pose_estimation:
            memory_per_frame *= 1.5  # Additional memory for pose processing
            
        if enable_team_classification:
            memory_per_frame *= 1.2  # Additional memory for crops and features
            
        optimal_batch = int(processing_memory_mb / memory_per_frame)

        # Basketball-specific batch size constraints
        if self.basketball_optimized:
            # For basketball, smaller batches often work better for team consistency
            optimal_batch = min(optimal_batch, 20)  # Cap at 20 for basketball
            
        # Apply general constraints
        optimal_batch = max(self.min_batch_size, min(optimal_batch, self.max_batch_size))

        self.logger.info(f"Basketball optimal batch size: {optimal_batch} "
                        f"(available: {available_memory_mb:.1f}MB, "
                        f"frame: {frame_memory_mb:.1f}MB, "
                        f"basketball_overhead: {basketball_overhead}MB)")

        return optimal_batch

    def optimize_batch_size(self,
                          frame_size: Tuple[int, int, int],
                          model_memory_mb: float = 500) -> int:
        """Legacy method - calls basketball optimization"""
        return self.optimize_basketball_batch_size(
            frame_size, model_memory_mb, 
            enable_pose_estimation=False,
            enable_team_classification=True
        )

    def adaptive_basketball_batch_generator(self,
                                          items: List,
                                          initial_batch_size: Optional[int] = None,
                                          player_count_hint: Optional[int] = None) -> Generator:
        """
        Generate batches with adaptive sizing for basketball processing

        Args:
            items: List of items to batch (frames or frame data)
            initial_batch_size: Initial batch size (None for auto)
            player_count_hint: Estimated player count for optimization

        Yields:
            Batches of items optimized for basketball processing
        """
        if initial_batch_size is None:
            current_batch_size = self.min_batch_size
        else:
            current_batch_size = initial_batch_size
            
        # Basketball-specific adaptations
        if self.basketball_optimized and player_count_hint:
            # Adjust initial batch size based on player count
            if player_count_hint > 8:  # Many players detected
                current_batch_size = max(1, int(current_batch_size * 0.8))
            elif player_count_hint < 4:  # Few players detected
                current_batch_size = min(self.max_batch_size, int(current_batch_size * 1.2))

        i = 0
        consecutive_errors = 0
        
        while i < len(items):
            # Get current batch
            batch_end = min(i + current_batch_size, len(items))
            batch = items[i:batch_end]

            # Yield batch and measure performance
            start_time = time.time()
            try:
                yield batch
                processing_time = time.time() - start_time
                consecutive_errors = 0  # Reset error count on success
                
                # Track basketball-specific metrics
                self._track_basketball_batch_metrics(batch, processing_time)
                
            except Exception as e:
                processing_time = time.time() - start_time
                consecutive_errors += 1
                self.logger.warning(f"Basketball batch processing error: {e}")
                
                # Reduce batch size on repeated errors
                if consecutive_errors >= 2:
                    current_batch_size = max(1, int(current_batch_size * 0.5))
                    self.basketball_stats['memory_optimizations_applied'] += 1
                    self.logger.info(f"Reduced batch size due to errors: {current_batch_size}")

            # Track performance
            self.processing_times.append(processing_time)
            self.batch_sizes.append(len(batch))

            # Adapt batch size based on basketball processing patterns
            if len(self.processing_times) >= 3:
                current_batch_size = self._adapt_basketball_batch_size(current_batch_size)

            i = batch_end

    def adaptive_batch_generator(self,
                               items: List,
                               initial_batch_size: Optional[int] = None) -> Generator:
        """Legacy method - calls basketball adaptive generator"""
        return self.adaptive_basketball_batch_generator(items, initial_batch_size)

    def _adapt_basketball_batch_size(self, current_size: int) -> int:
        """Adapt batch size based on basketball processing performance"""
        # Calculate recent performance metrics
        recent_times = self.processing_times[-5:]
        recent_sizes = self.batch_sizes[-5:]

        # Calculate processing rate (items/second)
        rates = [size / max(time_val, 0.001) for size, time_val in zip(recent_sizes, recent_times)]
        avg_rate = np.mean(rates)

        # Check memory usage
        memory_usage = psutil.virtual_memory().percent / 100

        # Basketball-specific adaptations
        new_size = current_size
        
        if memory_usage > 0.9:  # High memory usage
            new_size = int(current_size * 0.7)  # More aggressive reduction for basketball
            self.basketball_stats['memory_optimizations_applied'] += 1
            
        elif memory_usage < 0.6:  # Low memory usage
            if avg_rate > 8:  # Good performance with basketball processing
                new_size = int(current_size * 1.1)  # Conservative increase
            elif avg_rate < 2:  # Poor performance
                new_size = int(current_size * 0.9)  # Slight reduction
                
        # Basketball-specific constraints
        if self.basketball_optimized:
            # Keep batch sizes reasonable for team classification stability
            new_size = min(new_size, 24)  # Cap for basketball
            
            # Ensure minimum batch size for basketball analytics
            new_size = max(new_size, 2)

        # Apply general constraints
        new_size = max(self.min_batch_size, min(new_size, self.max_batch_size))

        if new_size != current_size:
            self.logger.info(f"Basketball batch adaptation: {current_size} -> {new_size} "
                           f"(rate: {avg_rate:.1f}/s, mem: {memory_usage:.1%})")

        return new_size

    def _adapt_batch_size(self, current_size: int) -> int:
        """Legacy method - calls basketball adaptation"""
        return self._adapt_basketball_batch_size(current_size)

    def _track_basketball_batch_metrics(self, batch: List, processing_time: float):
        """Track basketball-specific batch metrics"""
        batch_size = len(batch)
        
        # Estimate players processed (heuristic)
        estimated_players = batch_size * 8  # Assume ~8 players per frame average
        self.basketball_stats['total_players_processed'] += estimated_players
        
        # Track processing types (would be set by calling code)
        if hasattr(batch, 'contains_team_classification'):
            self.basketball_stats['team_classification_batches'] += 1
            
        if hasattr(batch, 'contains_pose_estimation'):
            self.basketball_stats['pose_estimation_batches'] += 1

    def get_basketball_performance_summary(self) -> Dict:
        """Get basketball-specific performance summary"""
        base_summary = self.get_performance_summary()
        
        # Add basketball-specific metrics
        basketball_summary = {
            **base_summary,
            'basketball_stats': self.basketball_stats.copy(),
            'basketball_optimized': self.basketball_optimized,
            'avg_players_per_batch': (self.basketball_stats['total_players_processed'] / 
                                     max(len(self.batch_sizes), 1)),
            'memory_optimization_rate': (self.basketball_stats['memory_optimizations_applied'] / 
                                       max(len(self.batch_sizes), 1))
        }
        
        return basketball_summary

    def get_performance_summary(self) -> Dict:
        """Get general performance summary"""
        if not self.processing_times:
            return {}

        return {
            'total_batches': len(self.processing_times),
            'avg_batch_size': np.mean(self.batch_sizes),
            'avg_processing_time': np.mean(self.processing_times),
            'total_processing_time': sum(self.processing_times),
            'avg_throughput': sum(self.batch_sizes) / max(sum(self.processing_times), 0.001)
        }

    def estimate_basketball_processing_time(self, 
                                          total_frames: int,
                                          frame_size: Tuple[int, int, int],
                                          enable_pose: bool = False,
                                          enable_team_classification: bool = True) -> Dict:
        """Estimate processing time for basketball analysis"""
        optimal_batch = self.optimize_basketball_batch_size(
            frame_size, enable_pose_estimation=enable_pose,
            enable_team_classification=enable_team_classification
        )
        
        # Estimate processing time per frame based on enabled features
        base_time_per_frame = 0.1  # Base detection + tracking
        
        if enable_team_classification:
            base_time_per_frame += 0.05  # Team classification overhead
            
        if enable_pose:
            base_time_per_frame += 0.1  # Pose estimation overhead
            
        # Basketball processing overhead
        base_time_per_frame += 0.02  # Basketball analytics
        
        estimated_total_time = total_frames * base_time_per_frame
        estimated_batches = int(np.ceil(total_frames / optimal_batch))
        
        return {
            'estimated_total_time_seconds': estimated_total_time,
            'estimated_total_time_minutes': estimated_total_time / 60,
            'optimal_batch_size': optimal_batch,
            'estimated_batches': estimated_batches,
            'estimated_fps': 1.0 / base_time_per_frame,
            'basketball_features': {
                'team_classification': enable_team_classification,
                'pose_estimation': enable_pose
            }
        }
