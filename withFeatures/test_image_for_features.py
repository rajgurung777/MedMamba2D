"""
Multi-task Learning: Predicts multiple outputs, which can include different types of 
predictions (e.g., classification and regression) from the same input.
The VSSMWithTwoHeads model you've designed fits into the multi-task learning category because 
it makes two independent predictions: one for classifying the input image and another for predicting 
additional features related to the input image.

Here since the model is trained with only the input image.
So accordingly for testing/inference, the input should only be an image and the output is a predicted (1) class and (2) features.
Note: Here all the additional features are taken together as a single prediction head (this is helpful when features are corelated)

Reference file: train_image_for_features.py

"""


import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from MedMamba import VSSM as medmamba

import torch.nn.functional as F


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


# Define transformations for the test images
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_class_indices(json_path):
    with open(json_path, 'r') as f:
        class_indices = json.load(f)
    return class_indices

# def predict(model, device, image_path, transform, class_indices):
#     """
#     Predict the class and additional features for a single image.

#     Args:
#     - model: The trained model.
#     - device: The device to perform computation on (CPU or CUDA).
#     - image_path: Path to the input image.
#     - transform: Transformations to be applied to the image.
#     - class_indices: Dictionary mapping class indices to class names.

#     Returns:
#     - class_pred: Predicted class.
#     - features_pred: Predicted additional features.
#     """
#     model.eval()
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         class_output, features_output = model(image)
#         class_pred = torch.argmax(class_output, dim=1).item()
#         features_pred = features_output.cpu().numpy().flatten()

#     class_name = [name for idx, name in class_indices.items() if idx == class_pred][0]

#     return class_name, features_pred


def predict(model, device, image_path, transform, class_indices):
    """
    Predict the class and additional features for a single image.

    Args:
    - model: The trained model.
    - device: The device to perform computation on (CPU or CUDA).
    - image_path: Path to the input image.
    - transform: Transformations to be applied to the image.
    - class_indices: Dictionary mapping class indices to class names.

    Returns:
    - class_pred: Predicted class name.
    - features_pred: Predicted additional features.
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        class_output, features_output = model(image)
        class_pred = torch.argmax(class_output, dim=1).item()
        features_pred = features_output.cpu().numpy().flatten()

    # Debugging: Print class_pred and class_indices
    print(f"class_pred (raw): {class_pred}")
    print(f"class_indices: {class_indices}")

    # Ensure the class_pred is of the same type as the keys in class_indices
    if isinstance(list(class_indices.keys())[0], str):
        class_pred = str(class_pred)
    else:
        class_pred = int(class_pred)

    # Retrieve the class name
    class_name = class_indices[class_pred]

    return class_name, features_pred



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    # Load the class indices
    class_indices = load_class_indices('class_indices.json')

    # Define the number of classes and additional features
    num_classes = len(class_indices)
    num_additional_features = 4  # Adjust as needed

    # Load the original model and the custom model with two heads
    original_model = medmamba(num_classes=num_classes)
    model = VSSMWithTwoHeads(original_model, num_classes=num_classes, num_additional_features=num_additional_features)
    
    # Load the trained model weights
    model.load_state_dict(torch.load('AmitImageNet.pth'))
    model.to(device)

    # Directory containing test images
    test_image_dir = 'images/dataset_B'  # Update with your test images directory

    # Predict for each image in the test directory
    for img_name in os.listdir(test_image_dir):
        img_path = os.path.join(test_image_dir, img_name)
        if os.path.isfile(img_path):
            class_pred, features_pred = predict(model, device, img_path, data_transform, class_indices)
            print(f"Image: {img_name}")
            print(f"Predicted Class: {class_pred}")
            print(f"Predicted Features: {features_pred}")
            print("-" * 30)









if __name__ == '__main__':
    main()

