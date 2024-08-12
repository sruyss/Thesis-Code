import numpy as np
import nibabel as nib
import pandas as pd

# This script creates separate segmentation masks for each set of coordinates
# in the provided CSV file. Each mask is saved as a NIfTI file.

def gaussian_distribution(size, center, sigma):
    """
    Generates a 3D Gaussian distribution centered at a given point.

    Args:
        size (tuple): The size of the output array (z, y, x).
        center (tuple): The (x, y, z) coordinates of the center.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: A 3D array with the Gaussian distribution.
    """
    y, x, z = np.meshgrid(np.arange(size[0]), np.arange(size[1]), np.arange(size[2]))
    d = np.linalg.norm(np.stack([x, y, z], axis=-1) - center, axis=-1)
    return np.exp(-d**2 / (2 * sigma**2))

def create_segmentation_mask(size, landmarks, sigma):
    """
    Creates segmentation masks for a set of landmarks using Gaussian distributions.

    Args:
        size (tuple): The size of the output mask (z, y, x).
        landmarks (numpy.ndarray): An array of landmark coordinates (x, y, z).
        sigma (float): The standard deviation for the Gaussian distribution.

    Returns:
        list: A list of 3D arrays, each representing a segmentation mask.
    """
    masks = []
    for landmark in landmarks:
        masks.append(gaussian_distribution(size, landmark, sigma))
    return masks

def save_masks(folder_name, masks, save_path):
    """
    Saves the generated segmentation masks as NIfTI files.

    Args:
        folder_name (str): The name of the folder where masks will be saved.
        masks (list): A list of 3D arrays representing the segmentation masks.
        save_path (str): The path where the NIfTI files will be saved.
    """
    for i, mask in enumerate(masks):
        mask_nifti = nib.Nifti1Image(mask, affine=None)
        mask_nifti_file_path = f"{save_path}/mask{i+1}.nii"
        nib.save(mask_nifti, mask_nifti_file_path)
        print(f"Mask {i+1} saved: {mask_nifti_file_path}")

# Path to the CSV file containing landmark coordinates
csv_path = '/home/sruyss8/Ultimate_coordinates.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Extract folder names and landmark coordinates from the DataFrame
folder_names = df['EAD']
landmarks = df[['TEA_X1', 'TEA_Y1', 'TEA_Z1', 'TEA_X2', 'TEA_Y2', 'TEA_Z2', 
                'PCA_X1', 'PCA_Y1', 'PCA_Z1', 'PCA_X2', 'PCA_Y2', 'PCA_Z2']].values

# Define the size of the segmentation masks and the standard deviation for the Gaussian
size = (192, 192, 192)
sigma = 10

# Iterate over each folder name and corresponding set of landmarks
for folder_name, landmark_row in zip(folder_names, landmarks):
    folder_name = str(folder_name)
    folder_path = f"/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/NewMasks/{folder_name}"
    
    # Reshape the landmarks and create segmentation masks
    masks = create_segmentation_mask(size, landmark_row.reshape(-1, 3), sigma)
    
    # Save the segmentation masks as NIfTI files
    save_masks(folder_name, masks, folder_path)
