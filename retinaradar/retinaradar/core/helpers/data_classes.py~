import pydantic
from pathlib import Path
from numpydantic import NDArray
from enum import Enum
from datetime import datetime
from typing import Literal, Optional, Union, Dict, Any, List
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class Image(pydantic.BaseModel):

    """
    Pydantic Data Class for Storing Image Information.

    parameters:
      path : absolute path to the image file
      name : image file name
      extension : image file extension
      original_shape : original dimensions for the image file (h,w,c)
      target_shape : dimensions to reshape image to (h,w,c)
      base64_string : base64 string encoding image data
      file_size_bytes : file size (in bytes)
    """

    path : Optional[Path] = pydantic.Field(
        default = None,
        frozen = True,
        description = "absolute path to image file"
    )
    name : Optional[str] = pydantic.Field(
        default = None,
        description = "image filename (without path)"
    )
    extension : Optional[str] = pydantic.Field(
        default = None,
        pattern=r'^\.[a-zA-Z0-9]+$',  # Must start with dot
        description="image file extension, including dot (e.g., '.jpg')"
    )
    original_shape : Optional[tuple[int, int, int]] = pydantic.Field(
        default = None,
        description = "original image shape in (h,w,c) format"
    )
    target_shape : Optional[tuple[int, int, int]] = pydantic.Field(
        default = None,
        description = "target dimensions to reshape to in (h,w,c) format"
    )
    base64_string : str | None = pydantic.Field(
        default = None,
        description = "Base64 encoded image data"
    )
    size_in_bytes: Optional[int] = pydantic.Field(
        default=None,
        ge=0,
        description="File size in bytes"
    )

    @pydantic.field_validator('original_shape', 'target_shape')
    @classmethod
    def validate_shape(cls, v):
        if v is not None and len(v) != 3:
            raise ValueError("Shape must be a 3-tuple (height, width, channels)")
        if v is not None and any(dim <= 0 for dim in v):
            raise ValueError("All dimensions must be positive")
        return v



class Quality(pydantic.BaseModel):

    """
    Pydantic Data Class for Storing Image Quality Information

    parameters:
       artifacts : image contains artifacts [0 = yes, 1 = no]
       focus : image is sharp and focuses [0 = no, 1 = yes]
       illumination : image lighting and illumination sufficient [0 = no, 1 = yes]
       contrast : image contrast sufficient [0 = no, 1 = yes]
       field : image field-of-view and centering sufficient [0 = no, 1 = yes]
       usable : image quality is overall sufficient [0 = no, 1 = yes]
    """

    artifacts : Optional[bool] = pydantic.Field(
        default = None,
        description = "image contains artifacts [0 = yes, 1 = no]"
    )
    clarity : Optional[bool] = pydantic.Field(
        default = None,
        description = "image is sharp and focused [0 = no, 1 = yes]"
    )
    illumination : Optional[bool] = pydantic.Field(
        default = None,
        description = "image lighting and illumination sufficient [0 = no, 1 = yes]"
    )
    contrast : Optional[bool] = pydantic.Field(
        default = None,
        description = "image contrast sufficient [0 = no, 1 = yes]"
    )
    field : Optional[bool] = pydantic.Field(
        default = None,
        description = "image field-of-view and centering sufficient [0 = no, 1 = yes]"
    )
    usable : Optional[bool] = pydantic.Field(
        default = None,
        description = "image quality is overall sufficient [0 = no, 1 = yes]"
    )





class Laterality(pydantic.BaseModel):

    """
    Store Eye Laterality [left | right] Information

    parameters :
       left_or_right : [left | right] left or right eye?
    """

    left_or_right : Optional[str] = pydantic.Field(
        default = None,
        description = "[left | right] left or right eye?"
    )



class FundusImageType(pydantic.BaseModel):

    """
    Store Fundus Image Type [standard | widefield | ultrawidefield] Information

    parameters:
       standard_widefield_ultrawidefield : [standard | widefield | ultrawidefield]
    """

    standard_widefield_ultrawidefield : Optional[str] = pydantic.Field(
        default = None,
        description = "[standard | widefield | ultrawidefield]"
    )



class Labels(pydantic.BaseModel):

    """
    Pydantic Data Class for Storing Datapoint Label Information

    parameters :
       laterality : [Laterality | None] describes image laterality (left / OS or right / OD)
       quality : [Quality | None] describes image quality in terms of illumination, image_field, artifacts, and focus
       image_type : [str | None] describes image type [standard FOV | widefield | ultrawidefield]
    """

    laterality : Optional[Laterality] = pydantic.Field(
        default = None,
        description = "describes left | right side, inverted status"
    )
    quality : Optional[Quality] = pydantic.Field(
        default = None,
        description = "describes image quality in terms of illumination, image_field, artifacts, and focus"
    )
    fundus_image_type : Optional[FundusImageType] = pydantic.Field(
        default = None,
        description = "type of fundus image [standard FOV | widefield | ultrawidefield]"
    )





class Datapoint(pydantic.BaseModel):

    """
    Pydantic Data Class for All Information for Single Datapoint

    parameters:
       uuid : a unique id for the datapoint
       source : the dataset the datapoint belongs to
       image : the Image data object
       labels : the Labels data object
    """

    uuid : Optional[str] = pydantic.Field(
        frozen = True,
        description = "unique id for the datapoint"
    )
    source : Optional[str] = pydantic.Field(
        default = None,
        description = "the dataset the datapoint belongs to"
    )
    image : Optional[Image] = pydantic.Field(
        default = None,
        description = "Image data and metadata"
    )
    labels : Optional[Labels] = pydantic.Field(
        default = None,
        description = "Labels for the image"
    )

    onehot_encoded_array: Optional[NDArray] = pydantic.Field(
        default=None,
        description="The one-hot encoded multilabel array, computed by the parent dataset."
    )    

    @pydantic.model_validator(mode='after')
    def validate_datapoint(self):
        """Ensure datapoint has minimum required information"""
        if self.uuid is None and self.image is None:
            raise ValueError("Datapoint must have either uuid or image information")
        return self

    @pydantic.computed_field
    @property
    def multilabel_array(self) -> List[Any]:
        """A list representation of all the labels for this datapoint."""
        labels = self.labels
        
        # Get laterality and fundus image type, defaulting to None
        lat = labels.laterality.left_or_right if labels and labels.laterality else None
        fit = labels.fundus_image_type.standard_widefield_ultrawidefield if labels and labels.fundus_image_type else None

        # Unpack all quality fields in a consistent order, defaulting to None
        quality_labels = [None] * len(Quality.model_fields)
        if labels and labels.quality:
            quality_values = labels.quality.model_dump()
            quality_labels = [quality_values.get(field) for field in Quality.model_fields]

        return [lat, fit] + quality_labels
    

    
    
class DatasetRole(str, Enum):
    """Enumeration for dataset roles"""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    BENCHMARK = "benchmark"
    PREDICTION = "prediction"


class RetinaRadarDataset(pydantic.BaseModel):
    """
    Pydantic Data Class for Storing a Group of Related Datapoints

    parameters:
       name : unique dataset name
       role : what role the dataset serves [training | validation | test | benchmark | prediction]
       datapoints : the Datapoint objects
    """
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str] = pydantic.Field(
        default=None,
        description="unique dataset name"
    )
    role: DatasetRole = pydantic.Field(
        default=DatasetRole.TRAINING,
        description="[training | validation | test | benchmark | prediction]"
    )
    datapoints: List[Datapoint] = pydantic.Field(
        default=[]
    )
    created_date: Optional[datetime] = pydantic.Field(
        default=None,
        description="Dataset creation time"
    )

    @pydantic.computed_field
    @property
    def total_count(self) -> int:
        """Total number of datapoints in the dataset"""
        return len(self.datapoints)

    @pydantic.computed_field
    @property
    def num_labels(self) -> Dict[str, int]:
        """Get the number of labels for each datapoint"""
        if not self.datapoints:
            return {}
        
        counts = {}
        for dp in self.datapoints:
            if not dp.labels:
                counts[dp.uuid] = 0
                continue

            count = 0
            label_groups = [dp.labels.laterality, dp.labels.quality, dp.labels.fundus_image_type]
            for group in label_groups:
                if group:
                    count += sum(1 for value in group.model_dump().values() if value is not None)
            counts[dp.uuid] = count
        return counts
    
    @pydantic.computed_field
    @property
    def onehot_encoder(self) -> Optional[OneHotEncoder]:
        """Create and fit a one-hot encoder for the labels based on all datapoints."""
        if not self.datapoints:
            return None

        # Use the new computed field from each datapoint to build the list for fitting
        all_labels_for_fitting = [dp.multilabel_array for dp in self.datapoints]

        if not all_labels_for_fitting:
            return None

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(all_labels_for_fitting)
        return encoder
    
    @pydantic.computed_field
    @property
    def multilabel_onehot_encoding(self) -> Dict[str, Optional[NDArray]]:
        """
        Provides the one-hot encoded array for each datapoint.
        Access a specific datapoint's array using its UUID as the key.
        e.g., dataset.multilabel_onehot_encoding[datapoint.uuid]
        """
        encoder = self.onehot_encoder
        if not self.datapoints or encoder is None:
            return {}

        encoded_data = {}
        for dp in self.datapoints:
            # Use the datapoint's own multilabel_array for the transformation
            encoded_data[dp.uuid] = encoder.transform([dp.multilabel_array])[0]
            
        return encoded_data

    @pydantic.computed_field
    @property
    def onehot_to_original_mapping(self) -> Dict[str, Any]:
        """Get a dictionary mapping feature names to their possible categorical values."""
        encoder = self.onehot_encoder
        if encoder is None:
            return {}

        feature_names = ['laterality', 'fundus_image_type'] + list(Quality.model_fields.keys())
        
        mapping = {}
        for i, feature in enumerate(feature_names):
            mapping[feature] = [cat for cat in encoder.categories_[i] if cat is not None]
        
        return mapping

    
    def compute_and_store_onehot_encodings(self) -> None:
        """
        Computes the one-hot encoding for each datapoint and stores
        the result directly in the datapoint's `onehot_encoded_array` field.
        """
        # This computed field already does the heavy lifting for all datapoints
        all_encodings = self.multilabel_onehot_encoding
        
        if not all_encodings:
            return

        for dp in self.datapoints:
            # Look up the pre-computed array and assign it to the field
            if dp.uuid in all_encodings:
                dp.onehot_encoded_array = all_encodings[dp.uuid]
    
    def add_datapoint(self, datapoint: Datapoint) -> None:
        """Add a datapoint to the dataset"""
        self.datapoints.append(datapoint)

    def get_by_uuid(self, uuid: str) -> Optional[Datapoint]:
        """Retrieve datapoint by UUID"""
        for dp in self.datapoints:
            if dp.uuid == uuid:
                return dp
        return None
