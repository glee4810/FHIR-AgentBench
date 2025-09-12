import json
import hashlib
import os
import pickle
import functools
from datetime import datetime, timedelta


# Cache configuration
CACHE_DIR = ".tool_cache"  # Directory where cache files are stored
CACHE_HOURS = None # 24    # Cache validity period in hours (None = never expire)
CACHE_ENABLED = True       # Global flag to enable/disable caching


def make_cache_key(tool_name, args, kwargs):
    """
    Creates a unique cache key from tool name and arguments.
    
    Example: get_patient("123", timeout=30) -> "abc123def456..."
    """
    # Create call information dictionary
    call_info = {
        'tool': tool_name,
        'args': args,
        'kwargs': kwargs
    }
    
    # Convert to JSON string (with fixed order)
    call_str = json.dumps(call_info, sort_keys=True, default=str)
    
    # Create short hash key
    return hashlib.md5(call_str.encode()).hexdigest()


def get_cache_file_path(cache_key):
    """Creates file path from cache key."""
    return os.path.join(CACHE_DIR, f"{cache_key}.cache")


def is_cache_fresh(file_path):
    """Checks if cache file is still valid."""
    if not os.path.exists(file_path):
        return False
    
    # If CACHE_HOURS is None, cache never expires
    if CACHE_HOURS is None:
        return True
    
    # Check file creation time
    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
    now = datetime.now()
    
    # Expired if older than configured hours
    return (now - file_time) < timedelta(hours=CACHE_HOURS)


def load_from_cache(tool_name, args, kwargs):
    """Loads result from cache. Returns None if not found."""
    # Create cache directory if it doesn't exist
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    # Create cache key and file path
    cache_key = make_cache_key(tool_name, args, kwargs)
    file_path = get_cache_file_path(cache_key)
    
    # Check if cache is valid
    if is_cache_fresh(file_path):
        try:
            # Read result from cache file
            with open(file_path, 'rb') as f:
                result = pickle.load(f)
                print(f"[CACHE HIT] Using cached result for {tool_name}({kwargs})")
                return result
        except Exception as e:
            print(f"[CACHE ERROR] Cannot read cache file: {e}")
            # Remove problematic cache file
            try:
                os.remove(file_path)
            except:
                pass
    
    return None


def save_to_cache(tool_name, args, kwargs, result):
    """Saves result to cache."""
    try:
        # Create cache directory if it doesn't exist
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        
        # Create cache key and file path
        cache_key = make_cache_key(tool_name, args, kwargs)
        file_path = get_cache_file_path(cache_key)
        
        # Save result to file
        with open(file_path, 'wb') as f:
            pickle.dump(result, f)
            print(f"[CACHE SAVE] Saved result for {tool_name}({kwargs}) to cache")
            
    except Exception as e:
        print(f"[CACHE ERROR] Failed to save cache: {e}")


def cached_tool(func):
    """
    Decorator that adds caching functionality to tool functions.
    
    Usage:
        @cached_tool
        def my_tool(arg1, arg2):
            # tool implementation
            return result
    """
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # If caching is disabled, just execute the function directly
        if not CACHE_ENABLED:
            return func(*args, **kwargs)
            
        tool_name = func.__name__
        
        # 1. Try to find result in cache
        cached_result = load_from_cache(tool_name, args, kwargs)
        if cached_result is not None:
            return cached_result
        
        # 2. Execute actual function if not in cache
        result = func(*args, **kwargs)
        
        # 3. Save result to cache
        save_to_cache(tool_name, args, kwargs, result)
        
        return result
    
    return wrapper


def clear_cache(all_files=False):
    """
    Cleans up cache files.
    
    Args:
        all_files: If True, delete all cache. If False, delete only expired cache.
    
    Returns:
        Number of deleted files
    """
    if not os.path.exists(CACHE_DIR):
        return 0
    
    deleted_count = 0
    
    # Check all .cache files in cache directory
    for filename in os.listdir(CACHE_DIR):
        if not filename.endswith('.cache'):
            continue
            
        file_path = os.path.join(CACHE_DIR, filename)
        
        # Delete all files or only expired files
        should_delete = all_files or not is_cache_fresh(file_path)
        
        if should_delete:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"[CACHE ERROR] Failed to delete file {filename}: {e}")
    
    if deleted_count > 0:
        print(f"[CACHE CLEANUP] Removed {deleted_count} cache files")
    
    return deleted_count


# Convenience functions
def clear_expired_cache():
    """Deletes only expired cache files."""
    return clear_cache(all_files=False)


def clear_all_cache():
    """Deletes all cache files."""
    return clear_cache(all_files=True)


def get_cache_info():
    """Returns cache information."""
    if not os.path.exists(CACHE_DIR):
        return {"total_files": 0, "total_size": 0}
    
    total_files = 0
    total_size = 0
    
    for filename in os.listdir(CACHE_DIR):
        if filename.endswith('.cache'):
            file_path = os.path.join(CACHE_DIR, filename)
            total_files += 1
            total_size += os.path.getsize(file_path)
    
    return {
        "total_files": total_files,
        "total_size": total_size,
        "cache_dir": os.path.abspath(CACHE_DIR)
    }