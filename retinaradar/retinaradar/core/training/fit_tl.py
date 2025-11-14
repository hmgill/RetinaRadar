"""
Improved FitTL with consolidated MLflow tracking and model artifacts
"""
import dill
import json 
import mlflow
import torch 
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from retinaradar.core.io import IO
from retinaradar.core.training.data_module import RetinaRadarDataModule
from retinaradar.core.models.tl_labeler import MultiLabelImageClassifier
from retinaradar.paths import PATHS


class FitTL:

    def __init__(self, config):
        self.config = config
        self.io = IO(config)

    def run(self):
        """
        CONFIGURE EXPERIMENT TRACKING WITH CONSOLIDATED OUTPUT
        """
        
        run_id = PATHS["run_id"]
        
        # Set MLflow tracking URI to store everything in the run-specific folder
        mlflow_tracking_uri = Path(PATHS["output"], "mlflow")
        mlflow_tracking_uri.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{mlflow_tracking_uri}")
        
        EXPERIMENT_NAME = f"Fit Transfer Learning Model | {run_id}"
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow_logger = MLFlowLogger(
            experiment_name=EXPERIMENT_NAME,
            tracking_uri=str(mlflow_tracking_uri)
        )

        with mlflow.start_run(run_id=mlflow_logger.run_id) as run:

            """
            READ DATASET
            """
            
            dataset = self.io.read_dataset()
            
            # Shuffle dataset
            dataset.shuffle(seed=42)
            print(dataset.get_shuffle_info())
            
            # Convert dataset to data module
            data_module = RetinaRadarDataModule(
                self.config,
                dataset
            )

            """
            INITIALIZE TL MODEL
            """

            # Get the number of labels
            num_labels = dataset.num_labels

            # Get model name
            model_name = self.config["tl"]["model"]["name"]
            
            # Get label names from the one-hot encoder
            label_names = dataset.onehot_encoder.get_feature_names_out().tolist()
            
            model = MultiLabelImageClassifier(
                num_labels=num_labels,
                model_name=model_name,
                learning_rate=self.config["tl"]["fit"]["hyperparameters"]["lr"]
            )

            """
            SAVE MODEL METADATA FOR INFERENCE (BEFORE TRAINING)
            """
            
            # Create artifacts directory within the run folder
            artifacts_dir = Path(PATHS["output"], "artifacts")
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract metadata from dataset
            metadata = dataset.get_metadata_for_inference()
            
            # Save metadata as JSON artifact in the run folder
            metadata_path = artifacts_dir / "label_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Log metadata to MLflow
            mlflow.log_artifact(str(metadata_path))
            mlflow.log_params({
                "num_labels": metadata['num_labels'],
                "model_name": model_name
            })
            mlflow.log_dict(metadata, "metadata/label_metadata.json")
            
            # Save the dataset object for reproducibility
            dataset_path = artifacts_dir / "dataset.dill"
            with open(dataset_path, 'wb') as f:
                dill.dump(dataset, f)
            mlflow.log_artifact(str(dataset_path))

            """
            SETUP MODEL CHECKPOINTING TO RUN FOLDER
            """
            
            # Create checkpoints directory within the run folder
            checkpoint_dir = Path(PATHS["output"], "checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup checkpoint callback to save best model
            checkpoint_callback = ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.4f}}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True
            )

            """
            TRAIN TL MODEL
            """
            
            trainer = pl.Trainer(
                max_epochs=self.config["tl"]["fit"]["hyperparameters"]["max_epochs"],
                accelerator="gpu",
                devices=1,
                logger=mlflow_logger,
                callbacks=[checkpoint_callback],
                default_root_dir=str(PATHS["output"])  # Set root dir for all PL outputs
            )
    
            trainer.fit(
                model,
                datamodule=data_module
            )

            """
            SAVE FINAL MODEL TO RUN FOLDER
            """
            
            # Create models directory within the run folder
            models_dir = Path(PATHS["output"], "models")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the trained PyTorch Lightning model
            final_model_path = models_dir / f"{model_name}_final.ckpt"
            trainer.save_checkpoint(str(final_model_path))
            
            # Log the final model to MLflow
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=f"{model_name}_{run_id}"
            )
            
            # Also save a state dict version for easier loading
            state_dict_path = models_dir / f"{model_name}_state_dict.pth"
            torch.save(model.state_dict(), str(state_dict_path))
            mlflow.log_artifact(str(state_dict_path))
            
            """
            CREATE INFERENCE PACKAGE
            """
            
            # Create a complete inference package with model + metadata
            inference_package = {
                'model_checkpoint': str(final_model_path),
                'metadata': metadata,
                'config': {
                    'model_name': model_name,
                    'num_labels': num_labels,
                    'label_names': label_names,
                    'learning_rate': self.config["tl"]["fit"]["hyperparameters"]["lr"]
                }
            }
            
            inference_package_path = artifacts_dir / "inference_package.json"
            with open(inference_package_path, 'w') as f:
                json.dump(inference_package, f, indent=2)
            
            mlflow.log_artifact(str(inference_package_path))
            
            # Log final summary
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"{'='*60}")
            print(f"Run ID: {run_id}")
            print(f"All outputs saved to: {PATHS['output']}")
            print(f"  - MLflow tracking: {mlflow_tracking_uri}")
            print(f"  - Checkpoints: {checkpoint_dir}")
            print(f"  - Models: {models_dir}")
            print(f"  - Artifacts: {artifacts_dir}")
            print(f"{'='*60}\n")
