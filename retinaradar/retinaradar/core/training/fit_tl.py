"""


"""
import dill
import json 
import mlflow 
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import MLFlowLogger

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
        CONFIGURE EXPERIMENT TRACKING 
        """
        
        run_id = PATHS["run_id"]
        
        EXPERIMENT_NAME = f"Fit Transfer Learning Model | {run_id}"
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow_logger = MLFlowLogger(experiment_name = EXPERIMENT_NAME)

        with mlflow.start_run(run_id = mlflow_logger.run_id) as run:

            """
            READ DATASET
            """
            
            dataset = self.io.read_dataset()
            # shuffle
            dataset.shuffle(seed=42)
            print(dataset.get_shuffle_info())

            '''
            dataset.compute_and_store_onehot_encodings()

            with open('dataset.dill', 'wb') as f:
                dill.dump(dataset, f)

            for datapoint in dataset.datapoints[:10]:
                l = datapoint.onehot_encoded_array

                l2 = dataset.decode_onehot_to_original(l)

                print(l)
                print(l2)
                print("\n")
            '''
            
            
            # convert dataset to data module
            data_module = RetinaRadarDataModule(
                self.config,
                dataset
            )


            """
            INITIALIZE TL MODEL
            """

            # get the number of labels
            num_labels = dataset.num_labels

            # get model name
            model_name = self.config["tl"]["model"]["name"]
            
            # get label names from the one-hot encoder
            label_names = dataset.onehot_encoder.get_feature_names_out().tolist()
            
            model = MultiLabelImageClassifier(
                num_labels = num_labels,
                model_name = model_name,
                learning_rate = self.config["tl"]["fit"]["hyperparameters"]["lr"],
                label_names = label_names
            )            


            """
            TRAIN TL MODEL
            """

            
            # --- 4. Training ---
            trainer = pl.Trainer(
                max_epochs = self.config["tl"]["fit"]["hyperparameters"]["max_epochs"],
                accelerator = "gpu",
                devices = 1, #"auto",
                logger = mlflow_logger
            )
    
            #trainer.fit(
            #    model,
            #    datamodule = data_module
            #)
            

            """
            SAVE MODEL METADATA FOR INFERENCE
            """
            # Extract metadata from dataset
            metadata = dataset.get_metadata_for_inference()
            
            # Save metadata as JSON artifact
            metadata_path = Path("label_metadata.json")
            with open(metadata_path, 'w+') as f:
                json.dump(metadata, f, indent=2)
            
            mlflow.log_artifact(str(metadata_path))
            
            # Also log as parameters for easy viewing in MLflow UI
            mlflow.log_param("num_labels", metadata['num_labels'])
            mlflow.log_param("feature_names", metadata['feature_names'])

            
            """
            SAVE TL MODEL ARTIFACT
            """

            '''
            # set registered model name
            registered_model_name = f"{model_name}_{run_id}"
            
            mlflow.pytorch.log_model(
                pytorch_model = model,
                registered_model_name = registered_model_name,
                artifact_path = PATHS["retinaradar_tl_model_efficientnet"] 
            )
            
            '''
