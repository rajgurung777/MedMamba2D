"""
Multi-task Learning: Predicts multiple outputs, which can include different types of 
predictions (e.g., classification and regression) from the same input.
The VSSMWithTwoHeads model you've designed fits into the multi-task learning category because 
it makes two independent predictions: one for classifying the input image and another for predicting 
additional features related to the input image.

Here since the model is trained with only the input image.
So accordingly for inference, the input should be only an image and the output is a predicted (1) class and (2) features.
Here all the additional features are taken together as a single prediction head (this is helpful when features are corelated)

Reference file: test_image_for_features.py
"""

import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from MedMamba import VSSM as medmamba  # Import the model
from PIL import Image

import torch.nn.functional as F


def load_data(csv_file):
    """
    Load and preprocess data from a CSV file.
    
    Args:
    - csv_file (str): Path to the CSV file.
    
    Returns:
    - df (pd.DataFrame): Preprocessed DataFrame.
    """
    df = pd.read_csv(csv_file)
    df['localization'] = df['localization'].astype('category').cat.codes
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})
    df['age_group'] = df['age_group'].astype('category').cat.codes
    return df


class CustomImageDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for loading images and additional features.
    
    Args:
    - annotations_file (str): Path to the CSV file with annotations.
    - img_dir (str): Directory with all the images.
    - transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = load_data(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.img_labels['Muthukumar_Classes'] = self.label_encoder.fit_transform(self.img_labels['Muthukumar_Classes'])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset.
        
        Args:
        - idx (int): Index of the item.
        
        Returns:
        - image (Tensor): Transformed image tensor.
        - additional_features (Tensor): Additional features tensor.
        - label (int): Encoded class label.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 3])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, -1]
        
        # Additional features
        localization = self.img_labels.iloc[idx, 4]
        larger_size = self.img_labels.iloc[idx, 5]
        gender = self.img_labels.iloc[idx, 6]
        age_group = self.img_labels.iloc[idx, 7]
        
        additional_features = torch.tensor([localization, larger_size, gender, age_group], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, additional_features, label


class VSSMWithTwoHeads(nn.Module):
    """
    Model with two output heads: one for class prediction and one for additional features prediction.
    
    Args:
    - original_model (nn.Module): The backbone model.
    - num_classes (int): Number of output classes for class prediction.
    - num_additional_features (int): Number of additional features.
    """
    def __init__(self, original_model, num_classes, num_additional_features):
        super(VSSMWithTwoHeads, self).__init__()
        self.original_model = original_model
        
        # Define the layers for class prediction
        self.fc1 = nn.Linear(768, 512)  # Input size is 768 (output of backbone), output size is 512
        self.fc2 = nn.Linear(512, num_classes)  # Output layer for class prediction
        
        # Define the layers for additional features prediction
        self.fc3 = nn.Linear(768, 512)  # Input size is 768 (output of backbone), output size is 512
        self.fc4 = nn.Linear(512, num_additional_features)  # Output layer for additional features prediction

    def forward(self, x):
        """
        Forward pass for the model.
        
        Args:
        - x (Tensor): Input image tensor.
        
        Returns:
        - class_output (Tensor): Predicted class logits.
        - features_output (Tensor): Predicted additional features.
        """
        # Forward pass through the backbone model
        x = self.original_model.forward_backbone(x)
        
        # Reshape the output tensor
        x = x.permute(0, 3, 1, 2)  # Permute dimensions to match expected input of the next layer
        x = self.original_model.avgpool(x)  # Apply average pooling
        x = torch.flatten(x, start_dim=1)  # Flatten the tensor to a 1D array
        
        # Head for class prediction
        x_class = F.relu(self.fc1(x))  # Apply ReLU activation to the first fully connected layer
        class_output = self.fc2(x_class)  # Compute class logits from the second fully connected layer
        
        # Head for additional features prediction
        x_features = F.relu(self.fc3(x))  # Apply ReLU activation to the first fully connected layer
        features_output = self.fc4(x_features)  # Compute additional features from the second fully connected layer
        
        return class_output, features_output  # Return both outputs



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = CustomImageDataset(annotations_file="images/dataset_B.csv", 
                                       img_dir="images/dataset_B", 
                                       transform=data_transform["train"])
    train_num = len(train_dataset)
    print("Number of training samples:", train_num)

    flower_list = train_dataset.img_labels['Muthukumar_Classes'].unique().tolist()
    cla_dict = {i: flower_list[i] for i in range(len(flower_list))}
    
    # Save class indices to a JSON file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # Number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    num_classes = len(flower_list)
    num_additional_features = 4  # Assuming there are 4 additional features
    original_model = medmamba(num_classes=num_classes)
    net = VSSMWithTwoHeads(original_model, num_classes=num_classes, num_additional_features=num_additional_features)
    net.to(device)
    
    class_loss_function = nn.CrossEntropyLoss()
    features_loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 100
    best_acc = 0.0
    save_path = './{}Net.pth'.format('AmitImage')
    train_steps = len(train_loader)

    train_loss_list = []
    train_acc_list = []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, features, labels = data
            optimizer.zero_grad()
            class_outputs, features_outputs = net(images.to(device))
            
            # Calculate loss for class prediction
            class_loss = class_loss_function(class_outputs, labels.to(device))
            
            # Calculate loss for additional features prediction
            features_loss = features_loss_function(features_outputs, features.to(device))
            
            # Total loss is the sum of both losses
            loss = class_loss + features_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(class_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        train_loss_list.append(running_loss / train_steps)
        train_acc = correct / total
        train_acc_list.append(train_acc)

        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, train_acc))

        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

    plt.figure()
    plt.plot(range(epochs), train_loss_list, label='Training Loss')
    plt.plot(range(epochs), train_acc_list, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
