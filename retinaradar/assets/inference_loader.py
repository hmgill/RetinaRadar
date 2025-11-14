"""
Inference Loader - Load trained models from consolidated run directories

This module provides utilities to load models and metadata from the 
consolidated run directory structure for inference.
"""

import json
import dill
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from retinaradar.core.models.tl_labeler import MultiLabelImageClassifier
from retinaradar.paths import get_inference_paths


class InferenceLoader:
    """
    Load models and metadata from a completed training run
    """
    
    def __init__(self, run_directory: Path):
        """
        Initialize the inference loader
        
        Args:
            run_directory: Path to the run directory (e.g., output/runs/run_XYZ-timestamp)
        """
        self.run_directory = Path(run_directory)
        self.inference_paths = get_inference_paths(self.run_directory)
        
        # Validate that the run directory exists
        if not self.run_directory.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_directory}")
    
    def load_inference_package(self) -> Dict[str, Any]:
        """
        Load the complete inference package
        
        Returns:
            dict: Complete inference package with model paths and metadata
        """
        package_path = self.inference_paths["inference_package"]
        
        if not package_path.exists():
            raise FileNotFoundError(f"Inference package not found: {package_path}")
        
        with open(package_path, 'r') as f:
            package = json.load(f)
        
        return package
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load label metadata for decoding predictions
        
        Returns:
            dict: Label metadata including feature names and categories
        """
        metadata_path = self.inference_paths["label_metadata"]
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def load_dataset(self):
        """
        Load the original dataset object
        
        Returns:
            RetinaRadarDataset: The dataset used for training
        """
        dataset_path = self.inference_paths["dataset"]
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        with open(dataset_path, 'rb') as f:
            dataset = dill.load(f)
        
        return dataset
    
    def load_model(
        self,
        checkpoint: str = "last",
        device: str = "cpu"
    ) -> MultiLabelImageClassifier:
        """
        Load the trained model
        
        Args:
            checkpoint: Which checkpoint to load ("last", "best", or specific path)
            device: Device to load model to ("cpu", "cuda", etc.)
            
        Returns:
            MultiLabelImageClassifier: Loaded model ready for inference
        """
        # Load inference package to get config
        package = self.load_inference_package()
        config = package['config']
        
        # Determine checkpoint path
        if checkpoint == "last":
            checkpoint_path = self.inference_paths["final_checkpoint"]
        elif checkpoint == "best":
            # Find best checkpoint by parsing checkpoint directory
            checkpoints_dir = self.run_directory / "checkpoints"
            checkpoints = sorted(checkpoints_dir.glob("*.ckpt"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
            # Assuming the best is saved with lowest val_loss in filename
            checkpoint_path = checkpoints[0]  # You may need to parse filenames to find truly best
        else:
            checkpoint_path = Path(checkpoint)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model from checkpoint
        model = MultiLabelImageClassifier.load_from_checkpoint(
            str(checkpoint_path),
            model_name=config['model_name'],
            num_labels=config['num_labels'],
            learning_rate=config['learning_rate'],
            label_names=config.get('label_names', [])
        )
        
        model = model.to(device)
        model.eval()
        
        return model
    
    def load_complete_inference_setup(
        self,
        checkpoint: str = "last",
        device: str = "cpu"
    ) -> Tuple[MultiLabelImageClassifier, Dict[str, Any], Any]:
        """
        Load everything needed for inference in one call
        
        Args:
            checkpoint: Which checkpoint to load
            device: Device to load model to
            
        Returns:
            tuple: (model, metadata, dataset)
        """
        model = self.load_model(checkpoint=checkpoint, device=device)
        metadata = self.load_metadata()
        dataset = self.load_dataset()
        
        return model, metadata, dataset
    
    def get_run_info(self) -> Dict[str, Any]:
        """
        Get information about the training run
        
        Returns:
            dict: Run information including paths and configuration
        """
        package = self.load_inference_package()
        metadata = self.load_metadata()
        
        info = {
            "run_directory": str(self.run_directory),
            "model_name": package['config']['model_name'],
            "num_labels": package['config']['num_labels'],
            "label_names": package['config']['label_names'],
            "feature_names": metadata['feature_names'],
            "inference_paths": {k: str(v) for k, v in self.inference_paths.items()}
        }
        
        return info
    
    def __repr__(self):
        return f"InferenceLoader(run_directory={self.run_directory})"


def quick_load_model(run_directory: Path, device: str = "cpu") -> Tuple:
    """
    Quick utility to load model and metadata for inference
    
    Args:
        run_directory: Path to the run directory
        device: Device to load model to
        
    Returns:
        tuple: (model, metadata, dataset)
        
    Example:
        >>> model, metadata, dataset = quick_load_model("output/runs/run_ABC-2025-01-15_120000")
        >>> predictions = model(images)
    """
    loader = InferenceLoader(run_directory)
    return loader.load_complete_inference_setup(device=device)


def decode_predictions(
    predictions: torch.Tensor,
    metadata: Dict[str, Any],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Decode model predictions back to original label format
    
    Args:
        predictions: Raw model output (logits) - shape: (num_labels,)
        metadata: Label metadata from training
        threshold: Threshold for binary classification
        
    Returns:
        dict: Decoded predictions with label names and human-readable format
        
    Example:
        >>> logits = model(image)
        >>> decoded = decode_predictions(logits[0], metadata)
        >>> print(decoded)  # Contains predictions with feature names as keys
    """
    # Apply sigmoid and threshold
    probabilities = torch.sigmoid(predictions)
    binary_predictions = (probabilities > threshold).float()
    
    # Get feature names (one-hot encoded feature names like 'x0_left', 'x1_standard', etc.)
    onehot_feature_names = metadata['onehot_feature_names']
    
    # Get original feature names (like 'laterality', 'fundus_image_type', etc.)
    feature_names = metadata['feature_names']
    
    # Get label categories for each original feature
    label_categories = metadata['label_categories']
    
    # Build a mapping from one-hot feature indices to (original_feature_idx, category_value)
    # This helps us understand which one-hot column corresponds to which original feature and value
    
    # Create the results dictionary
    results = {}
    
    # First, organize predictions by original feature
    feature_predictions = {fname: [] for fname in feature_names}
    
    for i, onehot_name in enumerate(onehot_feature_names):
        # Parse the one-hot feature name to extract original feature index and value
        # Format: "x{feature_idx}_{value}"
        # Example: "x0_left" means feature 0 (laterality) with value "left"
        
        if '_' in onehot_name:
            prefix, value = onehot_name.split('_', 1)
            feature_idx = int(prefix[1:])  # Extract number from 'x0', 'x1', etc.
            
            # Get the original feature name
            if feature_idx < len(feature_names):
                original_feature_name = feature_names[feature_idx]
                
                # Store the prediction for this specific value
                feature_predictions[original_feature_name].append({
                    'value': value,
                    'probability': float(probabilities[i]),
                    'prediction': bool(binary_predictions[i]),
                    'onehot_index': i
                })
    
    # Now process each original feature to get the best prediction
    for feature_name, predictions_list in feature_predictions.items():
        if not predictions_list:
            # No predictions for this feature
            results[feature_name] = {
                'probability': 0.0,
                'prediction': False,
                'label': None
            }
            continue
        
        # Find the value with highest probability
        best_pred = max(predictions_list, key=lambda x: x['probability'])
        
        results[feature_name] = {
            'probability': best_pred['probability'],
            'prediction': best_pred['prediction'],
            'label': best_pred['value'] if best_pred['prediction'] else None
        }
        
        # Also include all predictions for this feature (for debugging/analysis)
        results[f'{feature_name}_all'] = predictions_list
    
    return results
