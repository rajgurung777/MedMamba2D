"""
This script perform test using the trained model.

Here since the model is trained with the additional features to classify different classes (5 classes).
So for inference the input is image along with the additional features and the output is the class.

"""


import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import pandas as pd

from MedMamba import VSSM as medmamba  # import model

import torch.nn as nn


class VSSMWithFeatures(nn.Module):
    def __init__(self, original_model, num_additional_features, num_classes):
        super(VSSMWithFeatures, self).__init__()
        self.original_model = original_model
        self.fc1 = nn.Linear(768 + num_additional_features, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, features):
        x = self.original_model.forward_backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = self.original_model.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        
        x = torch.cat((x, features), dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Function to load the model
def load_model(model_path, num_additional_features, num_classes, device):
    original_model = medmamba(num_classes=num_classes)
    model = VSSMWithFeatures(original_model, num_additional_features=num_additional_features, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# Function to load class indices
def load_class_indices(json_file):
    with open(json_file, 'r') as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}

# # Function to preprocess the image and additional features
# def preprocess(image_path, additional_features, transform):
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image)
#     additional_features = torch.tensor(additional_features, dtype=torch.float32)
#     return image.unsqueeze(0), additional_features.unsqueeze(0)


def preprocess(image_path, additional_features, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    additional_features = torch.tensor(additional_features, dtype=torch.float32)
    return image.unsqueeze(0).to(device), additional_features.unsqueeze(0).to(device)




# Function to predict the class of a single image
# def predict(image_path, additional_features, model, class_indices, transform):
#     image, features = preprocess(image_path, additional_features, transform)
#     output = model(image, features)
#     _, predicted = torch.max(output, 1)
#     predicted_class = class_indices[predicted.item()]
#     return predicted_class, additional_features


def predict(image_path, additional_features, model, class_indices, transform, device):
    image, features = preprocess(image_path, additional_features, transform, device)
    output = model(image, features)
    _, predicted = torch.max(output, 1)
    predicted_class = class_indices[predicted.item()]
    return predicted_class, additional_features




# Main function to test the model
def main():
    model_path = 'AmitNewNet.pth'
    json_file = 'class_indices.json'
    num_additional_features = 4
    num_classes = 5  # Update this with the correct number of classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Load the model and class indices
    model = load_model(model_path, num_additional_features, num_classes, device)
    class_indices = load_class_indices(json_file)

    # Image path and additional features for testing
    # image_path = 'images/dataset_B/0000.png'
    # image_path = 'images/dataset_B/0001.png'
    # image_path = 'images/dataset_B/0006.png'

    # image_path = 'images/dataset_B/0014.png'
    image_path = 'images/dataset_B/0022.png'
    # image_path = 'images/dataset_B/0028.png'



    additional_features = [0, 2.5, 0, 1]  # Example additional features [localization, larger_size, gender, age_group]

    # Transformation for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Predict the class and print the results
    predicted_class, additional_features = predict(image_path, additional_features, model, class_indices, transform, device)
    print(f"Predicted Class: {predicted_class}")
    print(f"additional_features: {additional_features}")
    print(f"Additional Features: Localization={additional_features[0]}, Larger Size={additional_features[1]}, Gender={additional_features[2]}, Age Group={additional_features[3]}")

if __name__ == '__main__':
    main()
