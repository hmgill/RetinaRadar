from pathlib import Path
from functools import wraps
from typing import Union, Callable, Any
import inspect

def validate_filepath(
    param_name: str = None,
    check_extension: bool = False,
    valid_extensions: set = None
):
    """
    Decorator to validate filepath parameter in any position.
    
    Args:
        param_name: Name of the parameter to validate (if None, assumes first param)
        check_extension: Whether to validate file extension
        valid_extensions: Set of valid extensions (e.g., {'.csv', '.xlsx'})
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature to find the filepath parameter
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Determine which parameter to validate
            if param_name:
                if param_name not in bound_args.arguments:
                    raise ValueError(f"Parameter '{param_name}' not found in function signature")
                filepath = bound_args.arguments[param_name]
            else:
                # Use first parameter
                first_param = list(bound_args.arguments.values())[1]
                filepath = first_param
            
            # Convert to Path and validate
            file_path = Path(filepath)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not file_path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
            
            # Check extension if specified
            if check_extension and valid_extensions:
                ext = file_path.suffix.lower()
                if ext not in valid_extensions:
                    raise ValueError(
                        f"Invalid extension '{ext}'. Valid: {sorted(valid_extensions)}"
                    )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
