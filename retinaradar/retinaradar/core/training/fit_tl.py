
"""


"""
import mlflow 
import pytorch_lightning as pl
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
            
            model = MultiLabelImageClassifier(
                num_labels = num_labels,
                model_name = model_name,
                learning_rate = self.config["tl"]["fit"]["hyperparameters"]["lr"]
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
    
            trainer.fit(
                model,
                datamodule = data_module
            )
            

            """
            SAVE TL MODEL ARTIFACT
            """

            # set registered model name
            registered_model_name = f"{model_name}_{run_id}"
            
            mlflow.pytorch.log_model(
                pytorch_model = model,
                registered_model_name = registered_model_name,
                artifact_path = PATHS["retinaradar_tl_model"] 
            )

            
