import cv2
import pathlib
import shortuuid
import time
import tqdm 
import pandas as pd
import numpy as np

from typing import Union, Optional

from retinaradar.core.helpers.decorators import * 
from retinaradar.core.helpers.data_classes import *



class Loader:

    
    def __init__(self):
        pass



    
    @validate_filepath(check_extension=True, valid_extensions={'.csv', '.tsv'})
    def read_df(
        self, 
        filename: Union[str, Path],
        index: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Function for reading a CSV or TSV file as a Pandas dataframe.
        parameters;
          filename [str | Path] : path to the file
          index [str] : column name to index the dataframe on
        """
        
        df = pd.read_csv(str(filename), sep=',')
        
        if index:
            df = df.set_index(index)
            
        return df


    def read_image(
        self,
        image_path:Union[str, Path],
        get_size_in_bytes:bool = False
    ):
        image = cv2.imread(str(image_path))

        if get_size_in_bytes:
            image_size_bytes = (image.size * image.itemsize)
            return image, image_size_bytes
    
        return image
            
    

    def create_dataset_object(
        self,
        data
    ):
        datapoints = []

        for d in tqdm.tqdm(data):

            """
            create datapoint image object
            """
            path = d["image_path"]
            print(d.items())
            extension = path.suffix
            name = path.stem
            this_image, size_in_bytes = self.read_image(path, get_size_in_bytes = True)
            original_shape = this_image.shape
            target_shape = (512, 512, 3) # hard-coded shape for labeling model
            base64_string = None # placeholder

            image = Image(**{
                "path" : path,
                "name" : name,
                "extension" : extension,
                "original_shape" : original_shape,
                "target_shape" : target_shape,
                "base64_string" : base64_string,
                "size_in_bytes" : size_in_bytes
            })


            """
            create datapoint laterality object
            """

            laterality = Laterality(**{
                "left_or_right" : d["left_or_right"]
            })


            """
            create datapoint quality object
            """

            quality = Quality(**{
                "illumination" : d["illumination"],
                "contrast" : d["contrast"],
                "clarity" : d["clarity"],
                "artifacts" : d["artifacts"],
                "field" : d["field"]
            })

            
            """
            create datapoint fundus image type object
            """

            fundus_image_type = FundusImageType(**{
                "standard_widefield_ultrawidefield" : d["fundus_image_type"]
            })

            
            """
            create datapoint labels object
            """

            labels = Labels(**{
                "laterality" : laterality,
                "quality" : quality,
                "fundus_image_type" : fundus_image_type
            })


            """
            create datapoint 
            """

            uuid = f"{image.name}_{shortuuid.uuid()}"
            source = d["source"]
            
            datapoint = Datapoint(**{
                "uuid" : uuid,
                "source" : source,
                "image" : image,
                "labels" : labels
            })

            # add datapoint to list of datapoints
            datapoints.append(datapoint)

        """
        create dataset
        """

        created_date = time.time()

        dataset = RetinaRadarDataset(**{
            "datapoints" : datapoints,
            "created_date" : created_date
        })

        return dataset 
        

            
    

            
    
    def match_images_and_metadata(
        self,
        filename : Union[str, Path]
    ):

        # retrieve image data
        image_data = self.read_df(filename)

        # get the map of dataset source : image paths
        source_metadata = self.read_df("source_image_paths.csv")

        source_path_dict = {
            x["source"] : list(pathlib.Path(x["path_to_images"]).glob("*")) for x in source_metadata.to_dict(orient='records')
        }

        all_source_info = []

        for source, paths in source_path_dict.items():

            name_path_dict = {x.stem:x for x in paths}

            source_df = image_data[image_data["source"] == source]

            source_df["image_path"] = source_df["name"].map(name_path_dict)

            source_info = source_df.to_dict(orient='records')

            all_source_info.append(source_info)

        all_source_info_flat = [y for x in all_source_info for y in x]

        return all_source_info_flat


            

    def read_dataset(
        self,
        filename: Union[str, Path],
            
    ):
        
        matched_info = self.match_images_and_metadata(filename)[:200]
        dataset = self.create_dataset_object(matched_info)

        return dataset 
        
        

    
if __name__ == "__main__":
    l = Loader()
    l.read_dataset("labels/merged_labels.csv")
    #df = io_handler.read_df("source_image_paths.csv")
