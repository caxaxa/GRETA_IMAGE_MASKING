# main.py
import os
from models.unet_model import compile_and_load_model
from utilities.file_management import  create_directory
from utilities.image_processing import process_and_zip_images

if __name__ == "__main__":
    # Set the base directory (modify as needed)
    BASE_DIR = os.getcwd()

    # Create directories if they do not exist

    create_directory(os.path.join(BASE_DIR, 'Inputs','images'))

    zip_folder_path = os.path.join(BASE_DIR, 'Inputs', 'raw_images//*.jpg')

    # # Unzip and load images (modify zip_path as needed)
    # unzip_images(zip_folder_path)

    # Compile and load the model
    model = compile_and_load_model(model_path=os.path.join(BASE_DIR, 'models/greta_model_300E_LATEST.h5'))

    # Process images (modify forecast_set_path as needed)
    process_and_zip_images(model, zip_folder_path, output_folder=os.path.join(BASE_DIR, 'Inputs', 'images'))
