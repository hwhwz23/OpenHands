"""Utilities for monitoring performance of actions."""

import time

from openhands.core.logger import openhands_logger as logger
from openhands.runtime.utils.system_stats import get_system_stats


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

            # Record start time and initial system stats
            start_time = time.time()
            get_system_stats()

            logger.info(f'Action execution started: {action_name}')

            # Execute the original function
            result = await func(*args, **kwargs)

            # Record end time and final system stats
            end_time = time.time()
            end_stats = get_system_stats()

            # Calculate execution time
            execution_time = end_time - start_time

            # Calculate resource usage differences
            cpu_percent = end_stats.get('cpu_percent', 0.0)
            memory_info = end_stats.get('memory', {})
            memory_usage_mb = memory_info.get('rss', 0) / (1024 * 1024)  # Convert to MB
            memory_percent = memory_info.get('percent', 0.0)

            # Log performance metrics
            logger.info(
                f'Action execution completed: {action_name}, '
                f'execution_time={execution_time:.3f}s, '
                f'cpu_percent={cpu_percent:.1f}%, '
                f'memory_usage={memory_usage_mb:.2f}MB ({memory_percent:.1f}%)'
            )

            return result

        def sync_wrapper(*args, **kwargs):
            # Get action name from the first argument (self) and second argument (action)
            action_name = 'unknown'
            if len(args) >= 2:
                action_name = getattr(args[1], 'action', 'unknown')

            # Record start time and initial system stats
            start_time = time.time()
            get_system_stats()

            logger.info(f'Action execution started: {action_name}')

            # Execute the original function
            result = func(*args, **kwargs)

            # Record end time and final system stats
            end_time = time.time()
            end_stats = get_system_stats()

            # Calculate execution time
            execution_time = end_time - start_time

            # Calculate resource usage differences
            cpu_percent = end_stats.get('cpu_percent', 0.0)
            memory_info = end_stats.get('memory', {})
            memory_usage_mb = memory_info.get('rss', 0) / (1024 * 1024)  # Convert to MB
            memory_percent = memory_info.get('percent', 0.0)

            # Log performance metrics
            logger.info(
                f'Action execution completed: {action_name}, '
                f'execution_time={execution_time:.3f}s, '
                f'cpu_percent={cpu_percent:.1f}%, '
                f'memory_usage={memory_usage_mb:.2f}MB ({memory_percent:.1f}%)'
            )

            return result

        # Return the appropriate wrapper based on whether the function is async or not
        if hasattr(func, '__code__') and 'await' in func.__code__.co_names:
            return async_wrapper
        return sync_wrapper
