import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize,Pad
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import glob
from PIL import Image
from torchvision.transforms.functional import pad


class SphericalImageRotationDataset(Dataset):
    """
    Custom PyTorch Dataset for loading spherical image pairs and their relative rotation vectors.
    """
    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform or Compose([
            Pad((10, 8, 11, 9)),#Resize((256, 256)),  # Adjust size as required by your model
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet weights
        ])

    # def pad_to_even(self, img):
    #     width, height = img.size
    #     new_width = width if width % 2 == 0 else width + 1
    #     new_height = height if height % 2 == 0 else height + 1
    #     padding = (0, 0, new_width - width, new_height - height)  # (left, top, right, bottom)
    #     return pad(img, padding)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]
        
        # Load and preprocess images
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label)

def extract_labels_from_filename(filename):
    base_filename = os.path.basename(filename)
    base_filename = os.path.splitext(base_filename)[0] 
    parts = base_filename.split('_')
    numeric_values = []
    for part in parts:
        try:
            numeric_values.append(float(part))
        except ValueError:
            pass 
    if len(numeric_values) >= 12:
        f1_label = numeric_values[:6]
        f2_label = numeric_values[6:12]
    else:
        print(f"Filename {filename} does not contain enough numeric values.")
        return None, None
    return np.array(f1_label), np.array(f2_label)

def create_transformation_matrix(labels):
    X, Y, Z = labels[0], labels[1], labels[2]
    roll, pitch, yaw = labels[5], labels[3], labels[4]

    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    def euler_to_rotation_matrix(roll, pitch, yaw):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        R = R_z @ R_y @ R_x
        return R

    R = euler_to_rotation_matrix(roll, pitch, yaw)

    t = np.array([[X],
                  [Y],
                  [Z]])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.squeeze(t)
    return T

def vector_from_rotation(R):
    R = R[:3, :3]
    trace_R = np.trace(R)
    theta = np.arccos((trace_R - 1) / 2)

    if np.isclose(theta, 0):
        uu = np.array([1, 0, 0])
    elif np.isclose(theta, np.pi):
        idx = np.argmax(np.diag(R) + 1)
        uu = R[:, idx] / np.sqrt(2 * (1 + R[idx, idx]))
    else:
        uu = (1 / (2 * np.sin(theta))) * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])

    theta = np.degrees(theta)
    return theta, uu
# def generate_pairs_with_labels(image_paths):
#     """
#     Generate all possible pairs of images and their relative rotation vector labels.
#     """
#     pairs = []
#     labels = []
    
#     for i in range(len(image_paths)):
#         for j in range(i + 1, len(image_paths)):
#             # img1_path = image_paths[i]
#             # img2_path = image_paths[j]
#             filename = image_paths[i][17:-4]+"_"+image_paths[j][17:-4]
#             f1_label, f2_label = extract_labels_from_filename(filename)
          
#             # f1_label = extract_labels_from_filename(img1_path)[0]
#             # f2_label = extract_labels_from_filename(img2_path)[1]
            
#             if f1_label is None or f2_label is None:
#                 continue  # Skip invalid filenames
            
#             # Create transformation matrices
#             T1 = create_transformation_matrix(f1_label)
#             T2 = create_transformation_matrix(f2_label)
#             # print(T1)
#             # Compute relative rotation
#             T_relative = np.dot(np.linalg.inv(T1), T2)
#             R_relative = T_relative[:3, :3]
#             theta, u = vector_from_rotation(R_relative)
#             rotation_vector = theta * np.transpose(u)
            
#             pairs.append((image_paths[i], image_paths[j]))
#             labels.append(rotation_vector.astype(np.float32))
    
#     return pairs, labels
def generate_pairs_with_labels(image_paths, max_distance=20.0, min_distance=2.0):

    pairs = []
    labels = []
    
    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            filename = image_paths[i][17:-4]+"_"+image_paths[j][17:-4]
            f1_label, f2_label = extract_labels_from_filename(filename)
            
            if f1_label is None or f2_label is None:
                continue  # Skip invalid filenames
            
            # Get positions
            pos1 = np.array([f1_label[0], f1_label[1], f1_label[2]])
            pos2 = np.array([f2_label[0], f2_label[1], f2_label[2]])
            
            # Distance check only
            distance = np.linalg.norm(pos2 - pos1)
            if distance > max_distance or distance < min_distance:
                continue
            
            # Create transformation matrices
            T1 = create_transformation_matrix(f1_label)
            T2 = create_transformation_matrix(f2_label)
            
            # Compute relative rotation
            T_relative = np.dot(np.linalg.inv(T1), T2)
            R_relative = T_relative[:3, :3]
            theta, u = vector_from_rotation(R_relative)
            rotation_vector = theta * np.transpose(u)
            
            pairs.append((image_paths[i], image_paths[j]))
            labels.append(rotation_vector.astype(np.float32))
    
    return pairs, labels
# class SphericalImageRotationDataset(Dataset):
#     """
#     Custom PyTorch Dataset for loading spherical image pairs and their relative rotation vectors.
#     """
#     def __init__(self, pairs, labels, transform=None):
#         self.pairs = pairs
#         self.labels = labels
#         # self.transform = transform or Compose([
#         #     Resize((256, 256)),  # Adjust size as required by your model
#         #     ToTensor(),
#         #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet weights
#         # ])
    
#     def __len__(self):
#         return len(self.pairs)
    
#     def __getitem__(self, idx):
#         img1_path, img2_path = self.pairs[idx]
#         label = self.labels[idx]
        
#         # Load and preprocess images
#         img1 = Image.open(img1_path).convert("RGB")
#         img2 = Image.open(img2_path).convert("RGB")
#         # img1 = self.transform(img1)
#         # img2 = self.transform(img2)
        
#         return img1, img2, torch.tensor(label)

def plot_image_pair_with_label(img1_path, img2_path, label):
    """
    Plot an image pair with the corresponding rotation vector label.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Load and plot the first image
    img1 = Image.open(img1_path).convert("RGB")
    axs[0].imshow(img1)
    axs[0].axis('off')
    axs[0].set_title(f"Image 1: \n{os.path.basename(img1_path)}",fontsize=10,wrap=True)
    
    # Load and plot the second image
    img2 = Image.open(img2_path).convert("RGB")
    axs[1].imshow(img2)
    axs[1].axis('off')
    axs[1].set_title(f"Image 2: \n{os.path.basename(img2_path)}",fontsize=10,wrap=True)
    
    # Show the label below the plots
    plt.suptitle(f"Label (Rotation Vector): {label}", y=0.05, fontsize=10)
    plt.tight_layout()
    plt.show()

