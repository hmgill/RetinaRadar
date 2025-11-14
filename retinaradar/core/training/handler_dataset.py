import torch
from torch.utils.data import Dataset
import cv2


class HandlerDataset(Dataset):
    
    """
    A PyTorch Dataset that wraps our custom RetinaRadar Dataset.

    parameters: 
      retina_radar_dataset [RetinaRadarDataset] : Pydantic data class for storing fundus image data and metadata
      transform : the augmentations / transform operations to apply to an image
    """
    
    def __init__(self, retina_radar_dataset, transform=None):

        # precompute datapont multilabel onehot emcodings
        retina_radar_dataset.compute_and_store_onehot_encodings()

        # initialize 
        self.datapoints = retina_radar_dataset.datapoints        
        self.transform = transform

        
        
    def __len__(self):
        # Return the total number of samples
        return len(self.datapoints)

    
    def __getitem__(self, idx):
        # retrieve datapoint at the given index
        datapoint = self.datapoints[idx]

        # read image file 
        image_path = str(datapoint.image.path)
        image = cv2.imread(image_path)

        # cv2 reads image in BGR format by default
        # convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply transforms
        if self.transform:
            image = self.transform(image = image)["image"]
            
        # Get the pre-computed one-hot encoded label
        # Ensure the labels are float for BCEWithLogitsLoss
        labels = torch.tensor(datapoint.onehot_encoded_array, dtype=torch.float32)
        
        return image, labels
