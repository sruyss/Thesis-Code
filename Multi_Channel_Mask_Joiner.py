import os
import nibabel as nib
import numpy as np

# Iterates over a directory that contains folders with separate segmentation masks
# and combines them into a single multi-channel segmentation mask.

# Directory containing folders with segmentation masks
base_dir = '/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/NewMasks/'

# Iterate over each folder within the base directory
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    # Check if the current path is a directory
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_path}")
        
        # Initialize a list to store the segmentation masks
        masks = []
        
        # Iterate over each file in the current folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Check if the file is a NIfTI file
            if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                try:
                    print(f"Loading file: {file_path}")
                    
                    # Load the segmentation mask data
                    mask_data = nib.load(file_path).get_fdata()
                    masks.append(mask_data)
                    
                except Exception as e:
                    # Handle errors that occur while loading the file
                    print(f"Error loading file: {file_path}")
                    print(f"Error message: {str(e)}")
        
        # Combine the loaded masks into a single multi-channel image
        if masks:
            multi_channel_mask = np.stack(masks, axis=0)
            
            # Define the output file path for the multi-channel mask
            output_file = os.path.join(folder_path, 'multi_channel_mask.nii')
            
            try:
                # Save the multi-channel mask as a new NIfTI file
                multi_channel_nifti = nib.Nifti1Image(multi_channel_mask, affine=None)
                nib.save(multi_channel_nifti, output_file)
                print(f"Multi-channel mask saved to: {output_file}")
                
            except Exception as e:
                # Handle errors that occur while saving the file
                print(f"Error saving multi-channel mask to: {output_file}")
                print(f"Error message: {str(e)}")
        else:
            # Inform the user if no NIfTI files were found in the folder
            print("No NIfTI files found in the folder.")
