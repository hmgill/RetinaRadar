#!/usr/bin/env python
"""
Retina Radar entry point
"""
import argparse
from retinaradar.main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retina Radar')
    
    parser.add_argument(
        '--config',
        required=True,
        help='Path to the configuration file'
    )


    """
    run cleanup
    """
    cleanup_group = parser.add_argument_group('cleanup-run', 'options for removing runs')

    # Add mutually exclusive group for id vs all
    cleanup_exclusive = cleanup_group.add_mutually_exclusive_group(required=False)
    
    cleanup_exclusive.add_argument(
        '--cleanup-run-id', 
        metavar='ID',
        help='Remove a specific run by ID'
    )

    cleanup_exclusive.add_argument(
        '--cleanup-run-all', 
        action='store_true',
        help='Remove all runs'
    )
    
    args = parser.parse_args()
    main(args)
