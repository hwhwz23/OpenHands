"""Utilities for monitoring performance of actions."""

import statistics
import threading
import time
from typing import Any, Optional

from openhands.core.logger import openhands_logger as logger
from openhands.runtime.utils.system_stats import get_system_stats


class ResourceSampler:
    """A class to sample system resources in the background."""

    def __init__(self, sample_interval: float = 1.0):
        """Initialize the resource sampler.

        Args:
            sample_interval: The interval between samples in seconds.
        """
        self.sample_interval = sample_interval
        self.samples: list[dict[str, Any]] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def start(self):
        """Start sampling system resources."""
        if self.running:
            return

        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        """Stop sampling system resources and return statistics.

        Returns:
            Dict containing statistics about the samples.
        """
        if not self.running:
            return {
                'cpu_percent': {'avg': 0.0, 'max': 0.0, 'samples': []},
                'memory_mb': {'avg': 0.0, 'max': 0.0, 'samples': []},
                'memory_percent': {'avg': 0.0, 'max': 0.0, 'samples': []},
                'sample_count': 0,
            }

        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)  # Wait for the thread to finish

        with self.lock:
            if not self.samples:
                return {
                    'cpu_percent': {'avg': 0.0, 'max': 0.0, 'samples': []},
                    'memory_mb': {'avg': 0.0, 'max': 0.0, 'samples': []},
                    'memory_percent': {'avg': 0.0, 'max': 0.0, 'samples': []},
                    'sample_count': 0,
                }

            # Extract CPU and memory values from samples
            cpu_values = [sample.get('cpu_percent', 0.0) for sample in self.samples]
            memory_mb_values = [
                sample.get('memory', {}).get('rss', 0) / (1024 * 1024)
                for sample in self.samples
            ]
            memory_percent_values = [
                sample.get('memory', {}).get('percent', 0.0) for sample in self.samples
            ]

            # Calculate statistics
            stats = {
                'cpu_percent': {
                    'avg': statistics.mean(cpu_values) if cpu_values else 0.0,
                    'max': max(cpu_values) if cpu_values else 0.0,
                    'samples': cpu_values,
                },
                'memory_mb': {
                    'avg': statistics.mean(memory_mb_values)
                    if memory_mb_values
                    else 0.0,
                    'max': max(memory_mb_values) if memory_mb_values else 0.0,
                    'samples': memory_mb_values,
                },
                'memory_percent': {
                    'avg': statistics.mean(memory_percent_values)
                    if memory_percent_values
                    else 0.0,
                    'max': max(memory_percent_values) if memory_percent_values else 0.0,
                    'samples': memory_percent_values,
                },
                'sample_count': len(self.samples),
            }

            return stats

    def _sample_loop(self):
        """Background thread that samples system resources."""
        while self.running:
            try:
                # Get system stats
                stats = get_system_stats()

                # Add to samples list
                with self.lock:
                    self.samples.append(stats)

                # Sleep for the specified interval
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.error(f'Error in resource sampler: {e}')
                time.sleep(self.sample_interval)  # Sleep and try again


class PerformanceMonitor:
    """A utility class to monitor performance metrics for actions."""

    @staticmethod
    def monitor_execution(func):
        """Decorator to monitor execution time and system resources for a function.

        Args:
            func: The function to monitor.

        Returns:
            A wrapped function that logs performance metrics.
        """

        async def async_wrapper(*args, **kwargs):
            # Get action name from the first argument (self) and second argument (action)
            action_name = 'unknown'
            if len(args) >= 2:
                action_name = getattr(args[1], 'action', 'unknown')

            # Record start time and create resource sampler
            start_time = time.time()
            sampler = ResourceSampler(sample_interval=0.5)  # Sample every 0.5 seconds
            sampler.start()

            logger.info(f'Action execution started: {action_name}')

            try:
                # Execute the original function
                result = await func(*args, **kwargs)
                return result
            finally:
                # Record end time and stop resource sampler
                end_time = time.time()
                resource_stats = sampler.stop()

                # Calculate execution time
                execution_time = end_time - start_time

                # Get statistics
                cpu_avg = resource_stats['cpu_percent']['avg']
                cpu_max = resource_stats['cpu_percent']['max']
                memory_avg_mb = resource_stats['memory_mb']['avg']
                memory_max_mb = resource_stats['memory_mb']['max']
                memory_avg_percent = resource_stats['memory_percent']['avg']
                memory_max_percent = resource_stats['memory_percent']['max']
                sample_count = resource_stats['sample_count']

                # Log performance metrics
                logger.info(
                    f'Action execution completed: {action_name}, '
                    f'execution_time={execution_time:.3f}s, '
                    f'samples={sample_count}, '
                    f'cpu_avg={cpu_avg:.1f}%, cpu_max={cpu_max:.1f}%, '
                    f'memory_avg={memory_avg_mb:.2f}MB ({memory_avg_percent:.1f}%), '
                    f'memory_max={memory_max_mb:.2f}MB ({memory_max_percent:.1f}%)'
                )

        def sync_wrapper(*args, **kwargs):
            # Get action name from the first argument (self) and second argument (action)
            action_name = 'unknown'
            if len(args) >= 2:
                action_name = getattr(args[1], 'action', 'unknown')

            # Record start time and create resource sampler
            start_time = time.time()
            sampler = ResourceSampler(sample_interval=0.5)  # Sample every 0.5 seconds
            sampler.start()

            logger.info(f'Action execution started: {action_name}')

            try:
                # Execute the original function
                result = func(*args, **kwargs)
                return result
            finally:
                # Record end time and stop resource sampler
                end_time = time.time()
                resource_stats = sampler.stop()

                # Calculate execution time
                execution_time = end_time - start_time

                # Get statistics
                cpu_avg = resource_stats['cpu_percent']['avg']
                cpu_max = resource_stats['cpu_percent']['max']
                memory_avg_mb = resource_stats['memory_mb']['avg']
                memory_max_mb = resource_stats['memory_mb']['max']
                memory_avg_percent = resource_stats['memory_percent']['avg']
                memory_max_percent = resource_stats['memory_percent']['max']
                sample_count = resource_stats['sample_count']

                # Log performance metrics
                logger.info(
                    f'Action execution completed: {action_name}, '
                    f'execution_time={execution_time:.3f}s, '
                    f'samples={sample_count}, '
                    f'cpu_avg={cpu_avg:.1f}%, cpu_max={cpu_max:.1f}%, '
                    f'memory_avg={memory_avg_mb:.2f}MB ({memory_avg_percent:.1f}%), '
                    f'memory_max={memory_max_mb:.2f}MB ({memory_max_percent:.1f}%)'
                )

        # Return the appropriate wrapper based on whether the function is async or not
        if hasattr(func, '__code__') and 'await' in func.__code__.co_names:
            return async_wrapper
        return sync_wrapper
