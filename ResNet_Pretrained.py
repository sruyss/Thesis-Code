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
)
from monai.data import ImageDataset, DataLoader, CacheDataset
from monai.utils import first
from monai.metrics import compute_hausdorff_distance
from monai.networks.nets import EfficientNet, AttentionUnet, ResNet, HighResNet
from sklearn.model_selection import train_test_split, KFold
from CustomResNet import CustomResNet

# ResNet Classification script that performs cross-validation with a pre-trained model


# Path to the CSV file containing image paths and labels
csv_path = '/home/sruyss8/Ultimate_coordinates.csv'
df = pd.read_csv(csv_path)

# Path to the folder containing DICOM images
folder_path = '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/'

# Column names for coordinates in the CSV file
coordinate_cols = [
    'TEA_X1', 'TEA_Y1', 'TEA_Z1', 
    'TEA_X2', 'TEA_Y2', 'TEA_Z2', 
    'PCA_X1', 'PCA_Y1', 'PCA_Z1', 
    'PCA_X2', 'PCA_Y2', 'PCA_Z2'
]

def load_data(df, coordinate_cols):
    """
    Load image paths and corresponding coordinates from the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels.
        coordinate_cols (list): List of column names for coordinates.

    Returns:
        tuple: Two lists - image paths and labels.
    """
    image_paths = [row['EAD_path'] for _, row in df.iterrows()]
    labels = torch.tensor(df[coordinate_cols].values, dtype=torch.float32)
    return image_paths, labels

# Load image paths and labels
image_paths, labels = load_data(df, coordinate_cols)

# Number of folds for cross-validation
num_folds = 5

# Create a KFold object to split the dataset into training and validation folds
kf = KFold(n_splits=num_folds, shuffle=True, random_state=1)

# Define the training data transformations
train_transforms = Compose([
    LoadImaged(keys=['image'], ensure_channel_first=True),  
    Spacingd(keys=['image'], pixdim=[0.8, 0.8, 0.8]),
    Transposed(keys=['image'], indices=[0, 2, 1, 3]),
    ResizeWithPadOrCropd(keys=['image'], spatial_size=(192, 192, 192)),
    ThresholdIntensityd(keys=['image'], threshold=0, above=True),
    ScaleIntensityd(keys=['image']),
    RandGaussianNoised(keys=['image'], prob=0.1),
    RandGaussianSharpend(keys=['image'], prob=0.1)
])

# Define the validation data transformations
val_transforms = Compose([
    LoadImaged(keys=['image'], ensure_channel_first=True),  
    Spacingd(keys=['image'], pixdim=[0.8, 0.8, 0.8]),
    Transposed(keys=['image'], indices=[0, 2, 1, 3]),
    ResizeWithPadOrCropd(keys=['image'], spatial_size=(192, 192, 192)),
    ThresholdIntensityd(keys=['image'], threshold=0, above=True),
    ScaleIntensityd(keys=['image'])
])

# Iterate over each fold
for fold, (train_index, val_index) in enumerate(kf.split(image_paths)):

    # Batch size for training and validation
    batch_size = 4

    # Get the training and validation data for the current fold
    train_paths_fold = [image_paths[i] for i in train_index]
    val_paths_fold = [image_paths[i] for i in val_index]
    train_labels_fold = labels[train_index]
    val_labels_fold = labels[val_index]

    # Create datasets and data loaders for the current fold
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

    # Model name and hyperparameters
    model_name = f'Res10Crossfold{fold}'
    learning_rate = 0.008
    weight_decay = 0
    sched_step_size = 100
    sched_gamma = 0.1

    # Define save path and TensorBoard writer
    base_save_path = '/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/Extended/'
    save_path = os.path.join(base_save_path, model_name)
    writer = SummaryWriter(log_dir=save_path)

    # Path to the pre-trained model
    state_dict_path = "/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/improvednetwork/MedicalNet_pytorch_files2/pretrain/resnet_10_23dataset.pth"
    
    # Load the pre-trained model state dictionary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))

    # Create the ResNet model
    model = ResNet(
        block='basic',
        layers=[1, 1, 1, 1],
        block_inplanes=[64, 128, 256, 512],
        conv1_t_stride=2,
        spatial_dims=3,
        n_input_channels=1,
        num_classes=12
    )

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
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
        'Extra_info': 'K-fold (46) rand test',
        'Validation_Set': val_paths_fold,
        'Preprocessing': {
            'Train_Transforms': train_transforms.transforms,
            'Val_Transforms': val_transforms.transforms
        }
    }

    experiment_data = []
    experiment_data.append(experiment_info)

    train_losses = []
    val_losses = []
    distance_means = {i: [] for i in range(4)}
    distance_stds = {i: [] for i in range(4)}

    # Training loop
    num_epochs = 150
    for epoch in range(num_epochs):
        model.train()
        train_batch_losses = []
        for batch_data in train_loader_fold:
            training_inputs = batch_data['image'].to(device)
            training_targets = batch_data['label'].to(device)
            optimizer.zero_grad()
            training_outputs = model(training_inputs)
            loss = criterion(training_outputs, training_targets)
            loss.backward()
            optimizer.step()
            train_batch_losses.append(loss.item() * training_inputs.size(0))

        epoch_train_loss = np.mean(train_batch_losses)
        train_losses.append(epoch_train_loss)

        # Validation loop
        model.eval()
        val_batch_losses = []
        distances = {i: [] for i in range(4)}
        with torch.no_grad():
            for val_data in val_loader_fold:
                val_inputs = val_data['image'].to(device)
                val_targets = val_data['label'].to(device)
                val_outputs = model(val_inputs)
                val_batch_losses.append(criterion(val_outputs, val_targets).item() * val_inputs.size(0))

                # Calculate distances between predicted and true coordinates
                for i in range(4):
                    pred_coords = val_outputs[:, i*3:(i+1)*3]
                    true_coords = val_targets[:, i*3:(i+1)*3]
                    batch_distances = (torch.sqrt(((pred_coords - true_coords) ** 2).sum(dim=1))) * 0.8
                    distances[i].extend(batch_distances.cpu().numpy())

        epoch_val_loss = np.mean(val_batch_losses)
        val_losses.append(epoch_val_loss)

        # Calculate mean and standard deviation of distances
        for i in range(4):
            distance_means[i].append(np.mean(distances[i]))
            distance_stds[i].append(np.std(distances[i]))

        # Calculate mean and standard deviation of losses
        train_loss_std = np.std(train_batch_losses)
        val_loss_std = np.std(val_batch_losses)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Loss Std: {train_loss_std:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Loss Std: {val_loss_std:.4f}")
        for i in range(4):
            print(f"Distance Mean Point {i + 1}: {distance_means[i][-1]:.4f}, Distance Std Point {i + 1}: {distance_stds[i][-1]:.4f}")

        # Learning rate scheduling step
        scheduler.step()

        # Add scalar values to TensorBoard
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/Train Std', train_loss_std, epoch)
        writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)
        writer.add_scalar('Loss/Validation Std', val_loss_std, epoch)
        for i in range(4):
            writer.add_scalar(f'Distance/Mean_Point_{i + 1}', distance_means[i][-1], epoch)
            writer.add_scalar(f'Distance/Std_Point_{i + 1}', distance_stds[i][-1], epoch)

        # Add histograms of model parameters
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
        # Add example images or other visualizations
        # writer.add_images('Images', images, epoch)

    # Save the model's state dictionary
    torch.save(model.state_dict(), f'/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/Extended/{model_name}_model_state_dict.pth')

    # Close the TensorBoard writer
    writer.close()

    # Store experiment information
    experiment_data.append({
        'Epoch': epoch + 1,
        'Train Loss': epoch_train_loss,
        'Val Loss': epoch_val_loss,
        'Distance Mean TEA1': distance_means[0][-1],
        'Distance Std TEA1': distance_stds[0][-1],
        'Distance Mean TEA2': distance_means[1][-1],
        'Distance Std TEA2': distance_stds[1][-1],
        'Distance Mean PCA1': distance_means[2][-1],
        'Distance Std PCA1': distance_stds[2][-1],
        'Distance Mean PCA2': distance_means[3][-1],
        'Distance Std PCA2': distance_stds[3][-1]
    })

    # Convert experiment data to pandas DataFrame
    experiment_df = pd.DataFrame(experiment_data)
    filename = f'/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/Extended/experiments_info.csv'

    # Check if the file exists and append new data
    if not os.path.isfile(filename):
        experiment_df.to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)
        df = pd.concat([df, experiment_df], ignore_index=True)
        df.to_csv(filename, index=False)
