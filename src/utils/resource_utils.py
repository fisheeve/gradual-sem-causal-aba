
import threading
import functools
import psutil


class MemoryUsageExceededException(Exception):
    """Custom exception to indicate that memory usage has exceeded a threshold."""
    pass


class TimeoutException(Exception):
    """Custom exception to indicate a timeout has occurred."""
    pass


def check_memory_usage():
    """Check the current memory usage of the system."""
    memory = psutil.virtual_memory()
    return memory.percent


def timeout(seconds):
    """Decorator to enforce a timeout on a function call."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutException(f"Function '{func.__name__}' timed out after {seconds} seconds")]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                raise TimeoutException(f"Function '{func.__name__}' timed out after {seconds} seconds")
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator
