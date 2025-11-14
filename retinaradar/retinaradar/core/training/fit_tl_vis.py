
"""


"""
import mlflow 
from pytorch_lightning.loggers import MLFlowLogger


from retinaradar.core.io import IO
from retinaradar.core.training.data_module import RetinaRadarDataModule
from retinaradar.paths import PATHS


import torch
import cv2
import numpy as np
import os
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def save_batch_previews(
    dataloader,
    pydantic_dataset,
    num_samples=8,
    output_dir="batch_previews",
    split_name="train"
):
    """
    Fetches one batch from a DataLoader, de-normalizes the images, decodes the labels,
    and saves image-label pairs to an output directory for verification.

    Args:
        dataloader (DataLoader): The DataLoader to sample from (e.g., train_dataloader).
        pydantic_dataset (Dataset): The Pydantic Dataset object used to build the dataloader.
                                    This is needed for the label mapping.
        num_samples (int): The number of samples to save from the batch.
        output_dir (str): The directory where preview images will be saved.
        split_name (str): A name for the split (e.g., 'train', 'val', 'test') for filenames.
    """
    print(f"Generating preview for '{split_name}' split...")
 
    # 1. Get the mapping from one-hot index to label name
    # The OneHotEncoder provides the names for each column in the correct order.
    label_names = pydantic_dataset.onehot_encoder.get_feature_names_out()

    # 2. Define the de-normalization transform
    # This reverses the normalization process to make images viewable.
    # The formula is: pixel = (pixel * std) + mean
    denormalize_transform = T.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD]
    )

    # 3. Fetch one batch of data
    images, labels = next(iter(dataloader))
    
    # Ensure we don't try to save more samples than are in the batch
    num_samples = min(num_samples, len(images))

    for i in range(num_samples):
        image_tensor = images[i]
        label_tensor = labels[i]

        # 4. De-normalize the image tensor
        img_denorm = denormalize_transform(image_tensor)
        
        # 5. Convert tensor to a NumPy array suitable for OpenCV
        # Permute from (C, H, W) to (H, W, C), clamp values to [0, 1], and scale to [0, 255]
        img_np = img_denorm.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        img_uint8 = (img_np * 255).astype(np.uint8)

        # **CRITICAL**: OpenCV expects BGR format, but our tensor is RGB. We must convert it.
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        # 6. Decode the one-hot encoded label tensor back to text
        active_indices = torch.where(label_tensor == 1.0)[0]
        decoded_labels = [label_names[idx] for idx in active_indices]
        label_text = ", ".join(decoded_labels)

        # 7. Write the decoded labels onto the image
        # We'll add a black rectangle background for better readability
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_bgr, (0, 0), (text_width + 10, text_height + 15), (0, 0, 0), -1)
        cv2.putText(
            img_bgr,
            label_text,
            (5, text_height + 5), # Position
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA
        )

        # 8. Save the final image
        filename = f"{split_name}_sample_{i}.png"
        cv2.imwrite(filename, img_bgr)
    
    print(f"Saved {num_samples} preview images to '{output_dir}/'")


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

            print("read as data module")
            
            data_module.setup('fit')  # Prepares train and val splits
            data_module.setup('test') # Prepares test split

            # 4. Get the dataloaders
            train_dl = data_module.train_dataloader()
            val_dl = data_module.val_dataloader()
            test_dl = data_module.test_dataloader()            

            save_batch_previews(train_dl, data_module.retina_radar_dataset, split_name="train")
            save_batch_previews(val_dl, data_module.retina_radar_dataset, split_name="validation")
            save_batch_previews(test_dl, data_module.retina_radar_dataset, split_name="test")
