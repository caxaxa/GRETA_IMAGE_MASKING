# Documentation for Image Processing and Zipping Script

## Overview
The script performs multiple tasks:
1. Load the trained model.
2. Load images from a given directory.
3. Process the images: predict masks, create overlays, and save the processed images.
4. Zip the processed images.
5. Write the image grouping information to a text file.

## Dependencies

- `google.colab`: To mount Google Drive to Google Colab.
- `tensorflow.keras.models`: For loading saved machine learning models.
- `glob`: For reading the files based on pattern matching.
- `os`: For directory and file path operations.
- `cv2`: OpenCV for image processing.
- `numpy`: For mathematical operations and handling arrays.
- `PIL.Image`: Python Imaging Library.
- `zipfile`: For zipping files.
- `shutil`: To copy file-related metadata.
- `matplotlib.pyplot`: For plotting (not used in this script).

## Steps

### 1. Unrip and Load the Images:
```python
load_and_unzio('path_to_zipfile')
```

### 2. Load the Pre-trained Model:
```python
from tensorflow.keras.models import load_model
model = load_model('/content/my_model_100IMG_300E_augmented.h5')
```

### 3. Define Paths and Patterns:

Here, we set the path to the folder containing the images and specify the pattern of the filenames we want to process.
```python
path = "/content/drive/My Drive/Termal_Inspec_Folder/TERMOGRAFIA CALDEIRAO/Termografia Caldeirao Grande/DJI_202306141327_053_CG3-SU5A/"
forecast_set_path = path + "DJI_*_T.JPG"
```

### 4. Initialize Variables:

- `zip_count`, `file_count`, `group_count`: Counters for zip files, individual files, and image groups.
- `image_count`: Total number of images to be processed.
- `group_size`: Size of each image group.
- `zip_n`: Number of images per zipfile.
- `zip_file`, `final_zip_file`: Zipfile objects for individual and final zip archives.
- `predicted_masks`: List to store the predicted masks.
- `image_groups`: List to store the details of image groups.

### 5. Process Images:

Iterate through each image and:
- Read the image.
- Predict the mask for the image using the trained model.
- Overlay the predicted mask on the original image.
- Save the masked image with a new filename.
- Add the masked image to the zip files.
- Update counters and print the progress.

### 6. Write Image Group Details:

After processing all images, the script writes the group details to a `.txt` file.
```python
with open("image_groups.txt", 'w') as file:
    file.writelines(image_groups)
```

### 7. Close the Zipfile:

After processing all images and writing them to the zip files, the final zip file (`final_zip_file`) is closed.
```python
final_zip_file.close()
```

### 8. Completion Message:
Print a message to indicate the completion of the processing.
```python
print("Processing complete.")
```

---

**Note:** The function `copy_metadata(img_path, new_image_path)` appears in the code but its definition is not provided in the given code. Ensure that this function is defined elsewhere in your complete code to handle copying metadata from the original image to the masked image.
