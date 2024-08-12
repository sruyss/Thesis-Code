import pandas as pd
import os
import datetime
import monai
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    Compose,
    ResizeWithPadOrCrop,
    ResizeWithPadOrCropd,
    Spacing,
    Spacingd,
    RandGaussianSmoothd,
    ThresholdIntensity,
    ThresholdIntensityd,
    LoadImaged,
    EnsureChannelFirst,
    Transpose,
    Transposed,
    RandGaussianNoised,
    RandAdjustContrastd,
    ScaleIntensity,
    ScaleIntensityd,
    RandGaussianSharpend,
    RandRotate90d,
    RandFlipd,
)
from monai.data import ImageDataset, DataLoader, CacheDataset
from monai.utils import first
from monai.metrics import compute_hausdorff_distance
from monai.networks.nets import EfficientNet, AttentionUnet, ResNet, HighResNet, UNet
from sklearn.model_selection import train_test_split, KFold
from CustomResNet import CustomResNet
import torchvision.models as models
from monai.losses import DiceLoss

# UNet segmentation script that performs cross-validation


# Path to the CSV file containing image paths and labels
# csv_path = 'C:/Users/ruyss/Downloads/Thesis/thesis/Crop_adjusted_landmark_coordinates_flipped.csv'
csv_path = '/home/sruyss8/Ultimate_coordinates.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Path to the folder containing DICOM images
# folder_path = 'C:/Users/ruyss/Downloads/Knee_rotation_new/DICOM/'
folder_path = '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/'

# Path to the folder containing segmentation masks
seg_folder_path = '/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/NewMasks20/'
mask = '/multi_channel_mask.nii'

def load_data(df, folder_path):
    """
    Load image paths and corresponding labels from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels.
        folder_path (str): Path to the folder containing DICOM images.

    Returns:
        tuple: Lists of image paths and label paths.
    """
    image_paths = [row['EAD_path'] for _, row in df.iterrows()]
    labels = [os.path.join(seg_folder_path, row['EAD']) for _, row in df.iterrows()]
    return image_paths, labels

# Load image paths and labels
image_paths, labels = load_data(df, folder_path)
labels = [s + mask for s in labels]

# Number of folds for cross-validation
num_folds = 5

# Create a KFold object for splitting the dataset into folds
kf = KFold(n_splits=num_folds, shuffle=True, random_state=1)

# Define transforms for training data
train_transforms = Compose([
    LoadImaged(keys=['image'], ensure_channel_first=True),
    LoadImaged(keys=['label'], ensure_channel_first=False),
    Spacingd(keys=['image'], pixdim=[0.8, 0.8, 0.8]),
    Transposed(keys=['image'], indices=[0, 2, 1, 3]),
    ResizeWithPadOrCropd(keys=['image'], spatial_size=(192, 192, 192)),
    ThresholdIntensityd(keys=['image'], threshold=0, above=True),
    ScaleIntensityd(keys=['image']),
    RandGaussianNoised(keys=['image'], prob=0.1),
    RandGaussianSharpend(keys=['image'], prob=0.1),
    RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=(0, 1)),
    RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0)
])

# Define transforms for validation data
val_transforms = Compose([
    LoadImaged(keys=['image'], ensure_channel_first=True),
    LoadImaged(keys=['label'], ensure_channel_first=False),
    Spacingd(keys=['image'], pixdim=[0.8, 0.8, 0.8]),
    Transposed(keys=['image'], indices=[0, 2, 1, 3]),
    ResizeWithPadOrCropd(keys=['image'], spatial_size=(192, 192, 192)),
    ThresholdIntensityd(keys=['image'], threshold=0, above=True),
    ScaleIntensityd(keys=['image'])
])

# Iterate over each fold for cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(image_paths)):

    batch_size = 4

    # Get the training and validation data for the current fold
    train_paths_fold = [image_paths[i] for i in train_index]
    val_paths_fold = [image_paths[i] for i in val_index]
    train_labels_fold = [labels[i] for i in train_index]
    val_labels_fold = [labels[i] for i in val_index]

    # Create train and validation datasets and data loaders for the current fold
    train_dataset_fold = CacheDataset(
        [{'image': img, 'label': label} for img, label in zip(train_paths_fold, train_labels_fold)],
        transform=train_transforms
    )
    val_dataset_fold = CacheDataset(
        [{'image': img, 'label': label} for img, label in zip(val_paths_fold, val_labels_fold)],
        transform=val_transforms
    )
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, num_workers=4)

    model_name = f'M25mask20fold{fold}'
    learning_rate = 0.003
    weight_decay = 0
    sched_step_size = 300
    sched_gamma = 0.1

    # Define paths for saving the model and TensorBoard logs
    base_save_path = '/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/UNet/'
    save_path = os.path.join(base_save_path, model_name)
    writer = SummaryWriter(log_dir=save_path)

    # Set the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
        channels=(4, 8, 16, 32, 64),
        strides=(2, 2, 2, 2),
        num_res_units=4,
        dropout=0.3
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    # Define loss function and optimizer
    criterion = DiceLoss(sigmoid=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=sched_step_size, gamma=sched_gamma)

    # Experiment configuration
    experiment_info = {
        'Experiment': model_name,
        'Model': model.module.__class__.__name__,
        'Optimizer': optimizer.__class__.__name__,
        'Learning_Rate': learning_rate,
        'Weight_Decay': weight_decay,
        'Scheduler_Step_Size': sched_step_size,
        'Scheduler_Gamma': sched_gamma,
        'Extra_info': 'Multi Channel Segmentation',
        'Validation_Set': val_paths_fold,
        'Preprocessing': {
            'Train_Transforms': train_transforms.transforms,
            'Val_Transforms': val_transforms.transforms
        }
    }

    experiment_data = [experiment_info]

    train_losses = []
    val_losses = []

    # Training loop
    num_epochs = 150
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_data in train_loader_fold:
            training_inputs = batch_data['image'].to(device)
            training_targets = batch_data['label'].to(device)
            optimizer.zero_grad()
            training_outputs = model(training_inputs)
            loss = criterion(training_outputs, training_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * training_inputs.size(0)
        epoch_loss /= len(train_loader_fold.dataset)
        train_losses.append(epoch_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data in val_loader_fold:
                val_inputs = val_data['image'].to(device)
                val_targets = val_data['label'].to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item() * val_inputs.size(0)
        val_loss /= len(val_loader_fold.dataset)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Learning rate scheduling step
        scheduler.step()

        # Log scalar values to TensorBoard
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        # Log histograms of model parameters
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

    # Save the model's state dictionary
    torch.save(model.state_dict(), f'{base_save_path}{model_name}_model_state_dict.pth')

    # Close the TensorBoard writer
    writer.close()

    # Convert experiment data to a pandas DataFrame
    experiment_df = pd.DataFrame(experiment_data)
    filename = f'{base_save_path}experiments_info.csv'

    # Check if the file exists
    if not os.path.isfile(filename):
        # Save the DataFrame to a CSV file with headers
        experiment_df.to_csv(filename, index=False)
    else:
        # Load the existing DataFrame, append new data, and save
        df = pd.read_csv(filename)
        df = pd.concat([df, experiment_df], ignore_index=True)
        df.to_csv(filename, index=False)
