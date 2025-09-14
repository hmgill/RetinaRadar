import pathlib
import json 
from confection import registry, Config
from loguru import logger

class ConfigReader():

    def __init__(self):

        self.config = dict()


    def read_config(self, config_path:str) -> dict:
        config_data = Config().from_disk(config_path)

        for key, value in config_data.items():
            if isinstance(value, str) and ("/" in value or "\\" in value):
                self.config[key] = pathlib.Path(value)
            elif isinstance(value, dict):
                self.config[key] = self.process_nested_dict(value)
            elif isinstance(value, list):
                self.config[key] = self.process_nested_list(value)
            else:
                self.config[key] = value

        # add config information to log
        config_str = json.dumps(
            self.config,
            indent=2,
            default=str,
            ensure_ascii=False
        )

        logger.info(f"config file loaded from: {config_path}")
        logger.info(f"config file contents:\n{config_str}")
        
        return self.config



    def process_nested_dict(self, nested_dict: dict) -> dict:
        result = {}
        for key, value in nested_dict.items():
            if isinstance(value, str) and ("/" in value or "\\" in value):
                result[key] = pathlib.Path(value)
            elif isinstance(value, dict):
                result[key] = self.process_nested_dict(value)
            elif isinstance(value, list):
                result[key] = self.process_nested_list(value)
            else:
                result[key] = value
        return result



    def process_nested_list(self, nested_list: list) -> list:
        result = []
        for item in nested_list:
            if isinstance(item, str) and ("/" in item or "\\" in item):
                result.append(pathlib.Path(item))
            elif isinstance(item, dict):
                result.append(self.process_nested_dict(item))
            elif isinstance(item, list):
                result.append(self.process_nested_list(item))
            else:
                result.append(item)
        return result
