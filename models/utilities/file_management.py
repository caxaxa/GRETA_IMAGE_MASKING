# file_management.py
import os
import zipfile

def unzip_images(zip_path, extract_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
