import os
import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import monai
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
from sklearn.model_selection import train_test_split
from CustomResNet import CustomResNet
import torchvision.models as models
from monai.losses import DiceLoss

# UNet segmentation script (single fold)

# Define CSV path and read the CSV file containing image paths and labels
csv_path = '/home/sruyss8/Final_coordinates.csv'
df = pd.read_csv(csv_path)

# Define folder paths containing DICOM images and segmentation masks
folder_path = '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/'
seg_folder_path = '/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/NewMasks/'

mask = '/multi_channel_mask.nii'

# Function to load DICOM images and corresponding coordinates
def load_data(df, folder_path):
    """
    Load image paths and labels from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels.
        folder_path (str): Directory containing the images.

    Returns:
        tuple: (image_paths, labels) where image_paths and labels are lists of paths.
    """
    image_paths = [os.path.join(folder_path, row['EAD']) for _, row in df.iterrows()]
    labels = [os.path.join(seg_folder_path, row['EAD']) for _, row in df.iterrows()]
    return image_paths, labels

# Load image paths and labels
image_paths, labels = load_data(df, folder_path)
new_list = [s + mask for s in labels]

# Split data into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, new_list, test_size=0.15, random_state=46
)

# Define training and validation transformations
train_transforms = Compose([
    LoadImaged(keys=['image'], ensure_channel_first=True),
    LoadImaged(keys=['label'], ensure_channel_first=False),
    Spacingd(keys=['image'], pixdim=[0.8, 0.8, 0.8]),
    Transposed(keys=['image'], indices=[0, 2, 1, 3]),
    ResizeWithPadOrCropd(keys=['image'], spatial_size=(160, 160, 160)),
    ThresholdIntensityd(keys=['image'], threshold=0, above=True),
    ScaleIntensityd(keys=['image']),
    RandGaussianNoised(keys=['image'], prob=0.1),
    RandGaussianSharpend(keys=['image'], prob=0.1),
    RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=(0, 1)),
    RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0)
])

val_transforms = Compose([
    LoadImaged(keys=['image'], ensure_channel_first=True),
    LoadImaged(keys=['label'], ensure_channel_first=False),
    Spacingd(keys=['image'], pixdim=[0.8, 0.8, 0.8]),
    Transposed(keys=['image'], indices=[0, 2, 1, 3]),
    ResizeWithPadOrCropd(keys=['image'], spatial_size=(160, 160, 160)),
    ThresholdIntensityd(keys=['image'], threshold=0, above=True),
    ScaleIntensityd(keys=['image'])
])

# Create datasets and data loaders
datadict = [{'image': img, 'label': label} for img, label in zip(train_paths, train_labels)]
datadict_val = [{'image': img, 'label': label} for img, label in zip(val_paths, val_labels)]

train_dataset = CacheDataset(datadict, transform=train_transforms)
val_dataset = CacheDataset(datadict_val, transform=val_transforms)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

# Define model and training parameters
model_name = 'M25mask30'
learning_rate = 0.003
weight_decay = 0
sched_step_size = 300
sched_gamma = 0.1

base_save_path = '/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/UNet/'
save_path = os.path.join(base_save_path, model_name)
writer = SummaryWriter(log_dir=save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=4,
    channels=(4, 8, 16, 32, 64),
    strides=(2, 2, 2, 2),
    num_res_units=4,
    dropout=0.3,
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
    'Validation_Set': val_paths,
    'Preprocessing': {
        'Train_Transforms': train_transforms.transforms,
        'Val_Transforms': val_transforms.transforms
    }
}

experiment_data = [experiment_info]
train_losses = []
val_losses = []

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_data in train_loader:
        training_inputs = batch_data['image'].to(device)
        training_targets = batch_data['label'].to(device)
        optimizer.zero_grad()
        training_outputs = model(training_inputs)
        loss = criterion(training_outputs, training_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * training_inputs.size(0)

    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Validation loop
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data['image'].to(device)
            val_targets = val_data['label'].to(device)
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_targets).item() * val_inputs.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Learning rate scheduling step
    scheduler.step()

    # Log values to TensorBoard
    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)

    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

    # Store experiment information
    # epoch_info = {
    #     'Epoch': epoch + 1,
    #     'Train Loss': epoch_loss,
    #     'Val Loss': val_loss
    # }
    # experiment_data.append(epoch_info)

# Save model state dictionary
torch.save(
    model.state_dict(),
    os.path.join(base_save_path, f'{model_name}_model_state_dict.pth')
)

# Close the TensorBoard writer
writer.close()

# Convert experiment data to pandas DataFrame and save to CSV
experiment_df = pd.DataFrame(experiment_data)
filename = os.path.join(base_save_path, 'experiments_info.csv')

if not os.path.isfile(filename):
    # Save the DataFrame to a CSV file with headers
    experiment_df.to_csv(filename, index=False)
else:
    # Load the existing DataFrame and append new data
    df = pd.read_csv(filename)
    df = pd.concat([df, experiment_df], ignore_index=True)
    df.to_csv(filename, index=False)
