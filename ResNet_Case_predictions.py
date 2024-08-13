import matplotlib
matplotlib.use('agg') 
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
from sklearn.model_selection import train_test_split
from CustomResNet import CustomResNet

# Read CSV file containing image paths and labels
# csv_path = 'C:/Users/ruyss/Downloads/Thesis/thesis/Crop_adjusted_landmark_coordinates_flipped.csv'
csv_path = '/home/sruyss8/Ultimate_coordinates.csv'
df = pd.read_csv(csv_path)

# Define folder path containing DICOM images
# folder_path = 'C:/Users/ruyss/Downloads/Knee_rotation_new/DICOM/'
folder_path = '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/'

# Define column names for coordinates
coordinate_cols = ['TEA_X1', 'TEA_Y1', 'TEA_Z1', 'TEA_X2', 'TEA_Y2', 'TEA_Z2', 'PCA_X1', 'PCA_Y1', 'PCA_Z1', 'PCA_X2', 'PCA_Y2', 'PCA_Z2']

# Define a function to load DICOM images and corresponding coordinates
def load_data(df, folder_path, coordinate_cols):
    image_paths = [row['EAD_path'] for _, row in df.iterrows()]
    labels = torch.tensor(df[coordinate_cols].values, dtype=torch.float32)
    return image_paths, labels

# Load image paths and labels
image_paths, labels = load_data(df, folder_path, coordinate_cols)

# Define transformations for data augmentation
train_transforms = Compose([
    LoadImaged(keys=['image'], ensure_channel_first=True),  
    Spacingd(keys=['image'], pixdim=[0.8,0.8,0.8]),
    Transposed(keys=['image'], indices=[0, 2, 1, 3]),
    ResizeWithPadOrCropd(keys=['image'], spatial_size=(192, 192, 192)),
    ThresholdIntensityd(keys=['image'], threshold=0, above=True),
    ScaleIntensityd(keys=['image'])
    # RandGaussianNoise(prob=0.2),
    # RandGaussianSmooth(),
    # RandAdjustContrast(),
    # RandGaussianSharpen                  
])


# Define cases to predict and their paths
fold0 = ['60758646', '70332671', '71979520', '72394182', '74242124', '74324385', '74506262', '75237982', '78721909', '79135984', '81073918', '83693093', '84086115', '84493667', '84673904', '86097409', '87508164', '87561445', '87632055', '88305875', '88601463', '89048227', '89145189_L', '89253017', '89838957']
fold1 = ['65742318', '70606777', '71464200', '72183247', '72946270', '72949589', '73381857', '73401697', '73933582', '74111725', '75664730', '75831941', '76410141', '76846179', '77279263', '77600781', '79807202_2', '81286437', '83289488', '84487073', '85299105', '87546768', '88284534', '88652938', '89938633']
fold2 = ['60994782', '70142914', '70710629', '71038442', '71134217', '71372833', '72805559', '73572398', '73812968', '73941205', '74657172', '75296749', '77334779', '78949211', '80847403', '86139458', '86220027', '87318648', '87398467', '87490587', '88052246', '88063995', '88369517', '89157473', '89614499']
fold3 = ['60005199', '60919489', '62763313', '63937577', '70092747', '70131974', '70861380', '71922256', '74369133', '77103729', '77461895', '77491348', '78006491', '79807202', '84284926', '84735679', '85267102', '86628518', '87382792', '87474482', '88635370', '88805445', '89645188', '89648455', '89885404']
fold4 = ['60219401', '61752064', '62689583', '64988662', '66314071', '68783570', '70193826', '70371737', '70674437', '71355861', '71466577', '71879662', '73143240', '74925850', '78175155', '80889405', '81058331', '82399825', '83187138', '84042936', '84498724', '87626008', '88316104', '88460316']

red_fold0 = ['60919489', '61752064', '63937577', '70092747', '71372833', '73381857', '78006491', '79807202', '82399825', '84487073', '87398467', '87546768', '88052246', '88063995', '88284534', '88652938', '89145189_L']
red_fold1 = ['71134217', '73933582', '74111725', '77600781', '80889405', '81073918', '83187138', '83289488', '85299105', '86628518', '87318648', '87474482', '87490587', '87626008', '88305875', '89157473', '89838957']
red_fold2 = ['60219401', '60758646', '65742318', '70142914', '70193826', '70861380', '71355861', '71466577', '73572398', '73812968', '74242124', '74324385', '76846179', '77279263', '81058331', '84086115', '85267102']
red_fold3 = ['68783570', '70371737', '70710629', '71464200', '72949589', '74657172', '75831941', '77491348', '78949211', '79807202_2', '83693093', '84498724', '84673904', '86097409', '86220027', '88601463']
red_fold4 = ['64988662', '70332671', '70606777', '71038442', '73143240', '73941205', '74369133', '74506262', '75664730', '78175155', '80847403', '84042936', '87561445', '88369517', '89885404', '89938633']

red_fold0_paths = ['/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/60919489', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/61752064', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/63937577', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/70092747', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/71372833', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/73381857', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/78006491', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/79807202', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/82399825', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/84487073', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/87398467', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/87546768', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/88052246', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/88063995', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/88284534', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/88652938', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/89145189_L']
red_fold1_paths = ['/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/71134217', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/73933582', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/74111725', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/77600781', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/80889405', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/81073918', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/83187138', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/83289488', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/85299105', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/86628518', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/87318648', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/87474482', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/87490587', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/87626008', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/88305875', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/89157473', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/89838957']
red_fold2_paths = ['/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/60219401', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/60758646', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/65742318', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/70142914', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/70193826', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/70861380', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/71355861', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/71466577', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/73572398', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/73812968', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/74242124', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/74324385', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/76846179', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/77279263', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/81058331', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/84086115', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/85267102']
red_fold3_paths = ['/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/68783570', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/70371737', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/70710629', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/71464200', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/72949589', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/74657172', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/75831941', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/77491348', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/78949211', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/79807202_2', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/83693093', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/84498724', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/84673904', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/86097409', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/86220027', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/88601463']
red_fold4_paths = ['/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/64988662', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/70332671', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/70606777', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/71038442', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/73143240', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/73941205', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/74369133', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/74506262', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/75664730', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/78175155', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/80847403', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/84042936', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/87561445', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/88369517', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/89885404', '/usr/local/micapollo01/MIC/DATA/STUDENTS/kkouko0/DATA/Knee_rotation/Knee_rotation_final/89938633']

model_name = 'ReducedRes50Ptfold4'
cases = red_fold4
paths = red_fold4_paths

# Load the saved model
model_path = f'/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/Extended/{model_name}_model_state_dict.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_state_dict = torch.load(model_path, map_location=device)

# Create an instance of the model
model = ResNet(
    block='bottleneck',
    layers=[3, 4, 6, 3],
    block_inplanes=[64, 128, 256, 512],
    conv1_t_stride=2,
    spatial_dims=3,
    n_input_channels=1,
    num_classes=12
)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.load_state_dict(model_state_dict)

# Send the model to the appropriate device
model.to(device)

results = []

for case in cases:
    # Find the string containing the substring and its index
    index = [ [i, s] for i, s in enumerate(image_paths) if case in s]

    train_labels = labels[index[0][0]]
    case_name = index[0][1]
    case_index = cases.index(case)

    train_paths = [paths[case_index]]

    datadict=[{'image':img, 'label':label} for img, label in zip(train_paths, train_labels)]

    # Create datasets and data loaders
    train_dataset = CacheDataset(datadict, transform=train_transforms)

    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    # Perform evaluation
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            inputs = data['image'].to(device)
            outputs = model(inputs)

    # image = train_dataset[0]['image']
    image = inputs.detach().cpu()
    label = train_labels
    image = image.squeeze()
    dot_size = 5

    TEA_X1 = label[0]
    TEA_Y1 = label[1]
    TEA_Z1 = label[2].item()
    TEA_Z1 = round(TEA_Z1)
    # TEA_Z1 = 196-TEA_Z1
    # print(TEA_Z1)
    TEA_X2 = label[3]
    TEA_Y2 = label[4]
    PCA_X1 = label[6]
    PCA_Y1 = label[7]
    PCA_X2 = label[9]
    PCA_Y2 = label[10]
    outputs = outputs.detach().cpu()
    print('Label:',label)
    print('Predicted:', outputs)
    TEA_X1_pred = outputs[0][0]
    TEA_Y1_pred = outputs[0][1]
    TEA_Z1_pred = outputs[0][2].item()
    TEA_Z1_pred = round(TEA_Z1_pred)
    TEA_X2_pred = outputs[0][3]
    TEA_Y2_pred = outputs[0][4]
    TEA_Z2_pred = outputs[0][5]
    PCA_X1_pred = outputs[0][6]
    PCA_Y1_pred = outputs[0][7]
    PCA_Z1_pred = outputs[0][8]
    PCA_X2_pred = outputs[0][9]
    PCA_Y2_pred = outputs[0][10]
    PCA_Z2_pred = outputs[0][11]

    result = {'EAD': case}
    result['TEA_X1'] = TEA_X1_pred.item()
    result['TEA_Y1'] = TEA_Y1_pred.item()
    result['TEA_Z1'] = TEA_Z1_pred
    result['TEA_X2'] = TEA_X2_pred.item()
    result['TEA_Y2'] = TEA_Y2_pred.item()
    result['TEA_Z2'] = TEA_Z2_pred.item()
    result['PCA_X1'] = PCA_X1_pred.item()
    result['PCA_Y1'] = PCA_Y1_pred.item()
    result['PCA_Z1'] = PCA_Z1_pred.item()
    result['PCA_X2'] = PCA_X2_pred.item()
    result['PCA_Y2'] = PCA_Y2_pred.item()
    result['PCA_Z2'] = PCA_Z2_pred.item()
    results.append(result)


    # Plot a point (dot) on the image
    plt.plot(TEA_X1, TEA_Y1, marker='o', markersize=dot_size, color='red')
    plt.plot(TEA_X2, TEA_Y2, marker='o', markersize=dot_size, color='red')
    plt.plot(PCA_X1, PCA_Y1, marker='o', markersize=dot_size, color='red')
    plt.plot(PCA_X2, PCA_Y2, marker='o', markersize=dot_size, color='red')

    plt.plot(TEA_X1_pred, TEA_Y1_pred, marker='o', markersize=dot_size, color='yellow')
    plt.plot(TEA_X2_pred, TEA_Y2_pred, marker='o', markersize=dot_size, color='yellow')
    plt.plot(PCA_X1_pred, PCA_Y1_pred, marker='o', markersize=dot_size, color='yellow')
    plt.plot(PCA_X2_pred, PCA_Y2_pred, marker='o', markersize=dot_size, color='yellow')


    # Plot the image
    plt.imshow(image[:,:,TEA_Z1_pred], cmap = 'gray')  # Assuming a grayscale image; adjust cmap as needed
    plt.savefig(f'/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/Predictions/Resnet/{model_name}/{case}_prediction.png')
    plt.close()
    # plt.show()


# Combine predictions and labels into a single dataframe
experiment_df = pd.DataFrame(results)
# Define output filename (change filename as needed)
filename = f'/usr/local/micapollo01/MIC/DATA/STUDENTS/sruyss8/Predictions/Resnet/{model_name}/{model_name}_Predictions.csv'


# Check if the file exists
if not os.path.isfile(filename):
    # Save the DataFrame to a CSV file with headers
    experiment_df.to_csv(filename, index=False)
else:
    # Load the existing DataFrame from the CSV file
    df = pd.read_csv(filename)
    # Append the new row to the DataFrame
    df = pd.concat([df, experiment_df], ignore_index=True)
    # Save the updated DataFrame to the CSV file
    df.to_csv(filename, index=False)