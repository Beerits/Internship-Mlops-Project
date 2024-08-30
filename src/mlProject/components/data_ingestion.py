import os
import urllib.request as request
import zipfile
from mlProject import logger
from mlProject.utils.common import get_size
from pathlib import Path
from mlProject.entity.config_entity import (DataIngestionConfig)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        # Initialize the DataIngestion class with a configuration object
        self.config = config

    
    def download_file(self):
        # Check if the file doesn't already exist
        if not os.path.exists(self.config.local_data_file):
            # Download the file from the source URL and save it locally
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            # Log the successful download with additional info
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            # Log that the file already exists and display its size
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        Extracts the zip file into the specified directory.
        The function returns None.
        """
        # Set the directory to extract the zip file
        unzip_path = self.config.unzip_dir
        # Ensure the unzip directory exists
        os.makedirs(unzip_path, exist_ok=True)
        # Open the zip file and extract all its contents into the directory
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
