"""
Retina Radar

Entry point for training and inference workflows.

Required args:
   --config : path to the Retina Radar config file

Optional args:
   --cleanup-run-id : clean up a specific run by ID
   --cleanup-run-all : clean up all runs
"""

import json
from pathlib import Path
from loguru import logger

from retinaradar.log import initialize_log
from retinaradar.config import ConfigReader
from retinaradar.cleanup import cleanup_id, cleanup_all
from retinaradar.core.training.fit_tl import FitTL
from retinaradar.infer import RetinaRadarInference


def run_inference(config: dict):
    """
    Run inference based on config settings
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting inference workflow")
    
    # Extract inference config
    inf_config = config.get("inference", {})
    input_config = inf_config.get("input", {})
    output_config = inf_config.get("output", {})
    
    # Validate required settings
    run_directory = inf_config.get("run_directory")
    if not run_directory:
        raise ValueError("inference.run_directory must be specified in config")
    
    run_directory = Path(run_directory)
    if not run_directory.exists():
        raise FileNotFoundError(f"Run directory not found: {run_directory}")
    
    # Initialize inference handler
    checkpoint = inf_config.get("checkpoint", "last")
    device = inf_config.get("device", "auto")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    threshold = inf_config.get("threshold", 0.5)
    batch_size = inf_config.get("batch_size", 32)
    
    logger.info(f"Initializing inference with:")
    logger.info(f"  Run directory: {run_directory}")
    logger.info(f"  Checkpoint: {checkpoint}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Threshold: {threshold}")
    logger.info(f"  Batch size: {batch_size}")
    
    inferencer = RetinaRadarInference(
        run_directory=run_directory,
        device=device,
        checkpoint=checkpoint
    )
    
    # Determine input mode and run inference
    single_image = input_config.get("single_image", "")
    batch_images = input_config.get("batch_images", "")
    directory = input_config.get("directory", "")
    
    results = None
    
    if single_image:
        # Single image mode
        logger.info(f"Running inference on single image: {single_image}")
        predictions = inferencer.predict(
            single_image,
            threshold=threshold,
            return_probabilities=True
        )
        results = {Path(single_image).name: predictions}
        
        if output_config.get("print_summary", True):
            print("\n" + "="*70)
            print(f"Image: {Path(single_image).name}")
            print("="*70)
            print(inferencer.get_summary(predictions))
            print("="*70)
    
    elif batch_images:
        # Batch mode from file list
        logger.info(f"Running inference on batch from: {batch_images}")
        
        # Read image paths from file
        with open(batch_images, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Found {len(image_paths)} images in batch file")
        
        predictions_list = inferencer.predict_batch(
            image_paths,
            threshold=threshold,
            batch_size=batch_size
        )
        
        results = {
            Path(img_path).name: pred 
            for img_path, pred in zip(image_paths, predictions_list)
        }
        
        if output_config.get("print_summary", True):
            print("\n" + "="*70)
            print(f"Processed {len(results)} images")
            print("="*70)
            for i, (filename, predictions) in enumerate(list(results.items())[:3]):
                print(f"\n{filename}:")
                print(inferencer.get_summary(predictions))
                if i == 2 and len(results) > 3:
                    print(f"\n... and {len(results) - 3} more images")
    
    elif directory:
        # Directory mode
        logger.info(f"Running inference on directory: {directory}")
        
        extensions = input_config.get("extensions", [".jpg", ".jpeg", ".png", ".bmp"])
        
        results = inferencer.predict_directory(
            directory=directory,
            extensions=extensions,
            threshold=threshold,
            batch_size=batch_size
        )
        
        logger.info(f"Processed {len(results)} images")
        
        if output_config.get("print_summary", True):
            print("\n" + "="*70)
            print(f"Processed {len(results)} images from {directory}")
            print("="*70)
            for i, (filename, predictions) in enumerate(list(results.items())[:3]):
                print(f"\n{filename}:")
                print(inferencer.get_summary(predictions))
                if i == 2 and len(results) > 3:
                    print(f"\n... and {len(results) - 3} more images")
    
    else:
        raise ValueError(
            "Must specify one of: inference.input.single_image, "
            "inference.input.batch_images, or inference.input.directory"
        )
    
    # Save outputs
    if results:
        # Save JSON
        json_file = output_config.get("json_file", "")
        if json_file:
            json_path = Path(json_file)
            logger.info(f"Saving predictions to JSON: {json_path}")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Save CSV
        csv_file = output_config.get("csv_file", "")
        if csv_file:
            csv_path = Path(csv_file)
            logger.info(f"Saving predictions to CSV: {csv_path}")
            save_predictions_to_csv(results, csv_path)
        
        # Save text summaries
        if output_config.get("save_text_summaries", False):
            summary_dir = Path(output_config.get("text_summary_dir", "prediction_summaries"))
            summary_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving text summaries to: {summary_dir}")
            
            for filename, predictions in results.items():
                summary_file = summary_dir / f"{Path(filename).stem}_summary.txt"
                with open(summary_file, 'w') as f:
                    f.write(f"Image: {filename}\n")
                    f.write("="*70 + "\n")
                    f.write(inferencer.get_summary(predictions))
                    f.write("\n")
    
    logger.info("✅ Inference complete!")


def save_predictions_to_csv(results: dict, csv_path: Path):
    """
    Save predictions to CSV file
    
    Args:
        results: Dictionary of predictions {filename: predictions}
        csv_path: Path to output CSV file
    """
    import pandas as pd
    
    rows = []
    for filename, predictions in results.items():
        row = {'filename': filename}
        
        for field, values in predictions.items():
            if field.startswith('raw_'):
                continue  # Skip raw probabilities/logits
            
            if isinstance(values, dict) and 'prediction' in values:
                row[f'{field}_probability'] = values['probability']
                row[f'{field}_prediction'] = values['prediction']
                row[f'{field}_label'] = values.get('label', '')
            else:
                row[field] = values
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} predictions to CSV")


def main(args):
    """
    Main entry point for RetinaRadar
    
    Args:
        args: Command line arguments
    """
    
    # Handle cleanup operations
    if hasattr(args, 'cleanup_run_all') and args.cleanup_run_all:
        logger.info("Cleaning up all runs")
        cleanup_all()
        return
    
    if hasattr(args, 'cleanup_run_id') and args.cleanup_run_id:
        logger.info(f"Cleaning up run: {args.cleanup_run_id}")
        cleanup_id(args.cleanup_run_id)
        return
    
    # Initialize log file
    initialize_log()
    
    # Read config file
    config = ConfigReader().read_config(args.config)
    
    # Determine mode
    mode_config = config.get("mode", {})

    """
    # Fit TL model
    if mode_config.get("fit", True):
        logger.info("Starting training workflow")
        print("="*70)
        print("TRAINING MODE")
        print("="*70)
        FitTL(config).run()
    """
    
    # Inference on TL model
    if mode_config.get("infer", True):
        logger.info("Starting inference workflow")
        print("="*70)
        print("INFERENCE MODE")
        print("="*70)
        run_inference(config)
    
