"""
Inference Script for RetinaRadar

This script loads a trained model and runs inference on user-provided images.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
from loguru import logger

from retinaradar.inference_loader import InferenceLoader, decode_predictions
from retinaradar.paths import get_inference_paths


class RetinaRadarInference:
    """
    Inference handler for RetinaRadar models
    """
    
    def __init__(
        self,
        run_directory: Union[str, Path],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint: str = "last"
    ):
        """
        Initialize the inference handler
        
        Args:
            run_directory: Path to the training run directory
            device: Device to run inference on ('cuda' or 'cpu')
            checkpoint: Which checkpoint to load ('last', 'best', or path to specific checkpoint)
        """
        self.run_directory = Path(run_directory)
        self.device = device
        
        logger.info(f"Initializing RetinaRadar Inference")
        logger.info(f"  Run directory: {self.run_directory}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Checkpoint: {checkpoint}")
        
        # Load model and metadata
        self.loader = InferenceLoader(self.run_directory)
        
        logger.info("Loading model...")
        self.model = self.loader.load_model(checkpoint=checkpoint, device=self.device)
        self.model.eval()
        
        logger.info("Loading metadata...")
        self.metadata = self.loader.load_metadata()
        
        # Setup preprocessing transforms (same as validation transforms)
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        
        logger.info("✅ Inference handler ready!")
        logger.info(f"  Model: {self.metadata['num_labels']} labels")
        logger.info(f"  Features: {', '.join(self.metadata['feature_names'])}")
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy.ndarray: Image in RGB format
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image with OpenCV (BGR format)
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess an image for inference
        
        Args:
            image: Image in RGB format (H, W, C)
            
        Returns:
            torch.Tensor: Preprocessed image tensor (1, C, H, W)
        """
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed["image"]
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        threshold: float = 0.5,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Run inference on a single image
        
        Args:
            image: Image path or numpy array (RGB format)
            threshold: Threshold for binary predictions
            return_probabilities: Whether to include raw probabilities in output
            
        Returns:
            dict: Predictions with decoded labels
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        
        # Preprocess
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits)
        
        # Move to CPU for processing
        logits = logits.cpu()
        probabilities = probabilities.cpu()
        
        # Decode predictions
        predictions = decode_predictions(logits[0], self.metadata, threshold=threshold)
        
        # Optionally include raw probabilities
        if return_probabilities:
            predictions['raw_probabilities'] = probabilities[0].numpy().tolist()
            predictions['raw_logits'] = logits[0].numpy().tolist()
        
        return predictions
    
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        threshold: float = 0.5,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Run inference on multiple images
        
        Args:
            images: List of image paths or numpy arrays
            threshold: Threshold for binary predictions
            batch_size: Batch size for inference
            
        Returns:
            list: List of prediction dictionaries
        """
        all_predictions = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Load and preprocess batch
            batch_tensors = []
            for img in batch_images:
                if isinstance(img, (str, Path)):
                    img = self.load_image(img)
                tensor = self.preprocess_image(img)
                batch_tensors.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(batch_tensor)
            
            logits = logits.cpu()
            
            # Decode each prediction in batch
            for j in range(len(batch_images)):
                predictions = decode_predictions(
                    logits[j], 
                    self.metadata, 
                    threshold=threshold
                )
                all_predictions.append(predictions)
        
        return all_predictions
    
    def predict_directory(
        self,
        directory: Union[str, Path],
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
        threshold: float = 0.5,
        batch_size: int = 32
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run inference on all images in a directory
        
        Args:
            directory: Path to directory containing images
            extensions: List of valid image extensions
            threshold: Threshold for binary predictions
            batch_size: Batch size for inference
            
        Returns:
            dict: Dictionary mapping image filenames to predictions
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"No images found in {directory}")
            return {}
        
        logger.info(f"Found {len(image_files)} images")
        
        # Run batch inference
        predictions_list = self.predict_batch(image_files, threshold=threshold, batch_size=batch_size)
        
        # Map filenames to predictions
        results = {
            img_file.name: pred 
            for img_file, pred in zip(image_files, predictions_list)
        }
        
        return results
    
    def get_summary(self, predictions: Dict[str, Any]) -> str:
        """
        Get a human-readable summary of predictions
        
        Args:
            predictions: Predictions dictionary from predict()
            
        Returns:
            str: Formatted summary
        """
        summary_lines = ["Predictions:"]
        
        for feature, values in predictions.items():
            if feature.startswith('raw_'):
                continue  # Skip raw probabilities/logits
            
            if isinstance(values, dict) and 'prediction' in values:
                pred = "✓" if values['prediction'] else "✗"
                prob = values['probability']
                label = values.get('label', 'N/A')
                summary_lines.append(f"  {feature}: {pred} (prob={prob:.3f}, label={label})")
            else:
                summary_lines.append(f"  {feature}: {values}")
        
        return "\n".join(summary_lines)


def main():
    """
    Example usage of the inference script
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="RetinaRadar Inference")
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Path to the training run directory'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to a single image file'
    )
    parser.add_argument(
        '--directory',
        type=str,
        help='Path to a directory of images'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='last',
        help='Checkpoint to load (last, best, or path to .ckpt file)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on (cuda or cpu)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Prediction threshold (default: 0.5)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for directory inference (default: 32)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path (optional)'
    )
    
    args = parser.parse_args()
    
    # Initialize inference handler
    inferencer = RetinaRadarInference(
        run_directory=args.run_dir,
        device=args.device,
        checkpoint=args.checkpoint
    )
    
    # Run inference
    if args.image:
        logger.info(f"\nProcessing single image: {args.image}")
        predictions = inferencer.predict(
            args.image,
            threshold=args.threshold,
            return_probabilities=True
        )
        
        print("\n" + "="*60)
        print(inferencer.get_summary(predictions))
        print("="*60)
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"Saved predictions to: {args.output}")
    
    elif args.directory:
        logger.info(f"\nProcessing directory: {args.directory}")
        results = inferencer.predict_directory(
            args.directory,
            threshold=args.threshold,
            batch_size=args.batch_size
        )
        
        print("\n" + "="*60)
        print(f"Processed {len(results)} images")
        print("="*60)
        
        # Print first few results
        for i, (filename, predictions) in enumerate(list(results.items())[:3]):
            print(f"\n{filename}:")
            print(inferencer.get_summary(predictions))
            if i == 2 and len(results) > 3:
                print(f"\n... and {len(results) - 3} more images")
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved predictions to: {args.output}")
    
    else:
        logger.error("Must provide either --image or --directory")
        parser.print_help()


if __name__ == "__main__":
    main()
