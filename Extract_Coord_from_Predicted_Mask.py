import os
import pandas as pd
import torch
import torch.nn as nn
import nibabel as nib
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    Compose, LoadImaged, Spacingd, Transposed, ResizeWithPadOrCropd,
    ThresholdIntensityd, ScaleIntensityd, RandGaussianNoised, RandGaussianSharpend,
    RandRotate90d, RandFlipd, SaveImaged, SaveImage
)
from monai.data import DataLoader, CacheDataset
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage

# Iterates over a directory that contains folders with predicted segmentation masks
# from a certain model and gets coordinates from this

def list_subfolders(folder_path):
    """
    List all subfolder names in a specified folder.
    
    Args:
        folder_path (str): The path to the folder.
    
    Returns:
        List[str]: A list of subfolder names.
    """
    # List all entries in the specified folder
    entries = os.listdir(folder_path)
    
    # Filter out only the subfolders
    subfolders = [entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))]
    
    return subfolders


def extract_largest_centroid(mask, threshold=0.5):
    """
    Extracts the centroid of the largest connected component from the mask.

    Args:
        mask (numpy.ndarray): The predicted mask.
        threshold (float): Threshold to convert mask to binary.

    Returns:
        tuple: (x, y, z) coordinates of the centroid of the largest connected component.
    """
    # Apply threshold to get binary mask
    binary_mask = mask > threshold

    # Label connected components
    labeled_mask, num_features = ndimage.label(binary_mask)

    if num_features == 0:
        # No connected components found, return the center of the mask
        nonzero_indices = np.transpose(np.nonzero(binary_mask))
        if len(nonzero_indices) == 0:
            # No nonzero voxels found, return the center of the mask
            center_x, center_y, center_z = np.array(mask.shape) // 2
            return np.array([center_x, center_y, center_z])
        
        # Calculate centroid from nonzero voxel coordinates
        centroid = np.mean(nonzero_indices, axis=0)

        return centroid

    # Find the size of each connected component
    component_sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_features + 1))

    # Identify the largest component
    largest_component = np.argmax(component_sizes) + 1

    # Find the centroid of the largest component
    centroid = ndimage.center_of_mass(binary_mask, labeled_mask, largest_component)

    return centroid

# Function to load data (image paths and labels)
def load_data(df, folder_path, seg_folder_path, mask):
    """
    Load image paths and labels from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing image metadata.
        folder_path (str): Path to the folder containing images.
        seg_folder_path (str): Path to the folder containing segmentation masks.
        mask (str): Mask file name to append to each label.

    Returns:
        tuple: A tuple containing lists of image paths and labels.
    """
    image_paths = [row['EAD_path'] for _, row in df.iterrows()]
    labels = [os.path.join(seg_folder_path, row['EAD'] + mask) for _, row in df.iterrows()]
    return image_paths, labels

# Path to the CSV file containing image paths and labels
csv_path = '/home/sruyss8/Ultimate_coordinates.csv'
# Load the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Model that made prediction identifier
model = 'M25mask30Redfold4'

# Define folder paths for images and segmentation masks
folder_path = f'/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/Predictions/UNet/{model}/'
seg_folder_path = '/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/NewMasks/'
mask = '/multi_channel_mask.nii'

# Load image paths and labels from the DataFrame
image_paths, labels = load_data(df, folder_path, seg_folder_path, mask)

# Get the list of subfolders in the image directory
subfolders = list_subfolders(folder_path)

# Substrings to search for (list of cases to process)
cases = subfolders
results = []

# Iterate over each case in the list of subfolders
for case in cases:
    # Find the index and string containing the case name
    index = [[i, s] for i, s in enumerate(image_paths) if case in s]

    # Get the corresponding label and path for training
    train_labels = labels[index[0][0]]
    train_label = [f'{train_labels}']
    case_name = index[0][1]
    train_paths = [case_name]
    
    print(train_paths)
    result = {'EAD': case}
    
    # Process segmentation masks for each case
    for idx in range(4, 0, -1):
        mask = nib.load(f'/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/Predictions/UNet/{model}/{case}/seg_channel_{idx}.nii').get_fdata()
        # Extract coordinates from the mask
        coordinates = extract_largest_centroid(mask, threshold=0.5)
        if idx == 1:
            result['PCA_X2'] = coordinates[0]
            result['PCA_Y2'] = coordinates[1]
            result['PCA_Z2'] = coordinates[2]
        elif idx == 2:
            result['PCA_X1'] = coordinates[0]
            result['PCA_Y1'] = coordinates[1]
            result['PCA_Z1'] = coordinates[2]
        elif idx == 3:
            result['TEA_X2'] = coordinates[0]
            result['TEA_Y2'] = coordinates[1]
            result['TEA_Z2'] = coordinates[2]
        elif idx == 4:
            result['TEA_X1'] = coordinates[0]
            result['TEA_Y1'] = coordinates[1]
            result['TEA_Z1'] = coordinates[2]

    # Append the result for each case to the results list
    results.append(result)

# Combine predictions and labels into a single DataFrame
experiment_df = pd.DataFrame(results)
# Define output filename for saving predictions
filename = f'/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/Predictions/UNet/{model}/Predictions.csv'

# Check if the output file already exists
if not os.path.isfile(filename):
    # Save the DataFrame to a CSV file with headers
    experiment_df.to_csv(filename, index=False)
else:
    # Load the existing DataFrame from the CSV file
    df = pd.read_csv(filename)
    # Append the new predictions to the existing DataFrame
    df = pd.concat([df, experiment_df], ignore_index=True)
    # Save the updated DataFrame back to the CSV file
    df.to_csv(filename, index=False)
