
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import glob
from imblearn.over_sampling import SMOTE
from torchsummary import summary
import time

class CustomDataset(Dataset):
    def __init__(self, image_paths, numeric_params, target_classes=None, transform=None):
        self.image_paths = image_paths
        self.numeric_params = numeric_params
        self.target_classes = target_classes
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        params = torch.tensor(self.numeric_params[idx], dtype=torch.float32)
        
        if self.target_classes is not None:
            target_class = torch.tensor(self.target_classes[idx], dtype=torch.long)
            return image, params, target_class

        return image, params

class ForwardModel(nn.Module):
    def __init__(self):
        super(ForwardModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8 + 3, 256)  # 64*8*8 is the output size after conv layers + 2 numeric params
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # Output size: 2 (binary classification)

    def forward(self, image, params):
        x = F.relu(self.conv1(image))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.cat((x, params), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def read_params(file_path):
    with open(file_path, 'r') as f:
        data = f.read().strip().split(',')
        string_param = data[0]
        float_params = [float(param) for param in data[1:]]
    return string_param, np.array(float_params)

# Initialize the model
forward_model = ForwardModel()
# summary(forward_model, [(3, 64, 64), (3,)])

# Use glob to find all image files
image_paths = glob.glob(r"E:\dataset_final_with sub\cap\cap\Sample_Input_Data\*.png")
param_files = glob.glob(r"E:\dataset_final_with sub\cap\cap\Sample_Input_Image\*.txt")


string_params = np.array([read_params(file)[0] for file in param_files])
for i in range(len(string_params)):
    if string_params[i] == 'silicon':
        string_params[i] = 1
    elif string_params[i] == 'silicon nitride':
        string_params[i] = 2
    else:
        string_params[i] = 3

string_params = np.reshape(string_params,(len(string_params),1))
# string_params = np.reshape(string_params,(len(string_params,1)))

numeric_params = np.array([read_params(file)[1] for file in param_files])
# numeric_params = np.hstack((numeric_params,string_params))
# numeric_params = numeric_params.astype(float)

# Read target classes from a CSV file
target_classes_df = pd.read_csv(r"E:\dataset_final_with sub\cap\cap\intersecsssstion_ - Copy.csv")
target_classes = target_classes_df['Intersection Found'].values
numeric_params_reshaped = numeric_params.reshape(len(numeric_params), -1)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(numeric_params_reshaped, target_classes)

# Map the resampled numeric parameters back to image paths
# Initialize an empty list to store resampled image paths
resampled_image_paths = []

# Iterate over each resampled sample
for resampled_sample in X_resampled:
    # Find the closest original sample (in terms of numeric parameters)
    closest_index = np.argmin(np.sum(np.abs(numeric_params_reshaped - resampled_sample), axis=1))
    # Append the corresponding image path to the resampled_image_paths list
    resampled_image_paths.append(image_paths[closest_index])

# Convert the list to a numpy array
resampled_image_paths = np.array(resampled_image_paths)


# image_paths = np.reshape(image_paths,(len(image_paths),1))
# combined = np.hstack((image_paths, numeric_params))
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(combined, target_classes)
# image_paths = X_resampled[:,0]
# numeric_params = X_resampled[:,1:]
# target_classes = y_resampled

# Split the data into training and validation sets
train_image_paths, val_image_paths, train_numeric_params, val_numeric_params, train_target_classes, val_target_classes = train_test_split(
    image_paths, numeric_params, target_classes, test_size=0.2, random_state=42)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Assuming images are resized to 64x64
    transforms.ToTensor(),
])

# Create datasets
train_dataset = CustomDataset(train_image_paths, train_numeric_params, target_classes=train_target_classes, transform=transform)
val_dataset = CustomDataset(val_image_paths, val_numeric_params, target_classes=val_target_classes, transform=transform)

# Create data loaders
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the optimizer and loss function
optimizer = optim.Adam(forward_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 20

# Model training
for epoch in range(num_epochs):
    forward_model.train()
    train_losses = []
    train_targets = []
    train_preds = []
    for images, params, target_classes in train_loader:
        optimizer.zero_grad()
        outputs = forward_model(images, params)
        loss = criterion(outputs, target_classes)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        train_targets.extend(target_classes.tolist())
        train_preds.extend(torch.argmax(outputs, dim=1).tolist())
    
    train_loss = np.mean(train_losses)
    train_accuracy = accuracy_score(train_targets, train_preds)
    train_precision = precision_score(train_targets, train_preds)
    train_recall = recall_score(train_targets, train_preds)
    train_f1 = f1_score(train_targets, train_preds)

    forward_model.eval()
    val_losses = []
    val_targets = []
    val_preds = []
    with torch.no_grad():
        for images, params, target_classes in val_loader:
            outputs = forward_model(images, params)
            loss = criterion(outputs, target_classes)
            
            val_losses.append(loss.item())
            val_targets.extend(target_classes.tolist())
            val_preds.extend(torch.argmax(outputs, dim=1).tolist())

    val_loss = np.mean(val_losses)
    val_accuracy = accuracy_score(val_targets, val_preds)
    val_precision = precision_score(val_targets, val_preds)
    val_recall = recall_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1 Score: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}")
