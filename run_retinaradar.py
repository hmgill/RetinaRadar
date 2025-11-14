"""
Retina Radar entry point

Supports both training and inference workflows through configuration files.

Examples:
    # Training
    ./retinaradar --config retinaradar_train.config
    
    # Inference
    ./retinaradar --config retinaradar_inference.config
    
    # Cleanup
    ./retinaradar --cleanup-run-all
    ./retinaradar --cleanup-run-id run_ABC123-2025-01-15_120000
"""

import argparse
from retinaradar.main import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Retina Radar - Multi-label retinal image classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        Training:
        %(prog)s --config retinaradar_train.config
  
        Inference:
        %(prog)s --config retinaradar_inference.config
  
        Cleanup:
        %(prog)s --cleanup-run-all
        %(prog)s --cleanup-run-id run_ABC123-2025-01-15_120000
        """
    )
    
    # Main argument
    parser.add_argument(
        '--config',
        type=str,
        help='Path to the configuration file'
    )
    
    # Cleanup arguments
    parser.add_argument(
        '--cleanup-run-all',
        action='store_true',
        help='Clean up all training runs'
    )
    
    parser.add_argument(
        '--cleanup-run-id',
        type=str,
        help='Clean up a specific training run by ID'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.config, args.cleanup_run_all, args.cleanup_run_id]):
        parser.error("Must provide --config, --cleanup-run-all, or --cleanup-run-id")
    
    main(args)
