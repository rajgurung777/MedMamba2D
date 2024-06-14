"""
Multi-task Learning: Predicts multiple outputs one for each features and class (all independently).

Predicts/Test a given input image: the ouput is one of the classes (among the five classes) and features like localization, larger_size, gender and age-group.

This is interesting, since we can now only input an image and we can predict the likely additional features of that image (even if the dataset does not have any prior features available). This is because while training we use the dataset where 

Reference file: train_image_for_features_4.py
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from MedMamba import VSSM as medmamba

class VSSMWithMultipleHeads(nn.Module):
    def __init__(self, original_model, num_classes, num_localizations, num_sizes, num_genders, num_age_groups):
        super(VSSMWithMultipleHeads, self).__init__()
        self.original_model = original_model
        
        self.fc_class = nn.Linear(768, 512)
        self.class_out = nn.Linear(512, num_classes)
        
        self.fc_loc = nn.Linear(768, 512)
        self.loc_out = nn.Linear(512, num_localizations)
        
        self.fc_size = nn.Linear(768, 512)
        self.size_out = nn.Linear(512, num_sizes)
        
        self.fc_gender = nn.Linear(768, 512)
        self.gender_out = nn.Linear(512, num_genders)
        
        self.fc_age = nn.Linear(768, 512)
        self.age_out = nn.Linear(512, num_age_groups)

    def forward(self, x):
        x = self.original_model.forward_backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = self.original_model.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        
        x_class = F.relu(self.fc_class(x))
        class_output = self.class_out(x_class)
        
        x_loc = F.relu(self.fc_loc(x))
        loc_output = self.loc_out(x_loc)
        
        x_size = F.relu(self.fc_size(x))
        size_output = self.size_out(x_size)
        
        x_gender = F.relu(self.fc_gender(x))
        gender_output = self.gender_out(x_gender)
        
        x_age = F.relu(self.fc_age(x))
        age_output = self.age_out(x_age)
        
        return class_output, loc_output, size_output, gender_output, age_output

def load_mappings():
    with open('feature_mappings.json', 'r') as json_file:
        mappings = json.load(json_file)
    return mappings

def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

def predict(model, device, image_path, transform):
    image = load_image(image_path, transform).to(device)
    model.eval()
    with torch.no_grad():
        class_pred, loc_pred, size_pred, gender_pred, age_pred = model(image)
        
        class_pred = class_pred.argmax(dim=1).item()
        loc_pred = loc_pred.argmax(dim=1).item()
        size_pred = size_pred.argmax(dim=1).item()
        gender_pred = gender_pred.argmax(dim=1).item()
        age_pred = age_pred.argmax(dim=1).item()
        # print("class_pred=",class_pred)

        return class_pred, loc_pred, size_pred, gender_pred, age_pred

def testSingle_image():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    mappings = load_mappings()
    
    # print("class_mapping:", class_mapping)
    
    num_classes = len(mappings['Muthukumar_Classes'])
    num_localizations = len(mappings['localization'])
    num_sizes = len(mappings['larger_size'])
    num_genders = len(mappings['gender'])
    num_age_groups = len(mappings['age_group'])
    
    original_model = medmamba(num_classes=num_classes)
    model = VSSMWithMultipleHeads(original_model, num_classes, num_localizations, num_sizes, num_genders, num_age_groups)
    model.load_state_dict(torch.load('./AmitMultiClassNet.pth'))
    model.to(device)

    test_image_path = 'images/dataset_B/0023.png'  # Replace with the path to your test image

    class_pred, loc_pred, size_pred, gender_pred, age_pred = predict(model, device, test_image_path, data_transform)

    # print("Predicted Class:", class_pred)
    # print("Predicted Localization:", loc_pred)
    # print("Predicted Size:", size_pred)
    # print("Predicted Gender:", gender_pred)
    # print("Predicted Age Group:", age_pred)

    # Reverse mappings
    class_mapping = {k: v for k, v in mappings['Muthukumar_Classes'].items()}   # Reverse mappings
    localization_mapping = {k: v for k, v in mappings['localization'].items()}
    size_mapping = {k: v for k, v in mappings['larger_size'].items()}
    gender_mapping = {k: v for k, v in mappings['gender'].items()}
    age_group_mapping = {k: v for k, v in mappings['age_group'].items()}

    # print("localization_mapping:", localization_mapping)
    # print("size_mapping:", size_mapping)
    # print("gender_mapping:", gender_mapping)
    # print("age_group_mapping:", age_group_mapping)

    print("Predicted Class:", class_mapping[str(class_pred)])
    print("Predicted Localization:", localization_mapping[str(loc_pred)])
    print("Predicted Size:", size_mapping[str(size_pred)])
    print("Predicted Gender:", gender_mapping[str(gender_pred)])
    print("Predicted Age Group:", age_group_mapping[str(age_pred)])

def testImage_Folder():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))


    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # Load the class indices
    mappings = load_mappings()
    
    num_classes = len(mappings['Muthukumar_Classes'])
    num_localizations = len(mappings['localization'])
    num_sizes = len(mappings['larger_size'])
    num_genders = len(mappings['gender'])
    num_age_groups = len(mappings['age_group'])
    
    original_model = medmamba(num_classes=num_classes)
    model = VSSMWithMultipleHeads(original_model, num_classes, num_localizations, num_sizes, num_genders, num_age_groups)
    model.load_state_dict(torch.load('./AmitMultiClassNet.pth'))
    model.to(device)


    # Directory containing test images
    test_image_dir = 'images/dataset_B'  # Update with your test images directory


    # Reverse mappings
    class_mapping = {k: v for k, v in mappings['Muthukumar_Classes'].items()}   # Reverse mappings
    localization_mapping = {k: v for k, v in mappings['localization'].items()}
    size_mapping = {k: v for k, v in mappings['larger_size'].items()}
    gender_mapping = {k: v for k, v in mappings['gender'].items()}
    age_group_mapping = {k: v for k, v in mappings['age_group'].items()}


    # Predict for each image in the test directory
    for img_name in os.listdir(test_image_dir):
        img_path = os.path.join(test_image_dir, img_name)
        if os.path.isfile(img_path):            

            class_pred, loc_pred, size_pred, gender_pred, age_pred = predict(model, device, img_path, data_transform)

            print(f"Image: {img_name}, Class: {class_mapping[str(class_pred)]}, Localization: {localization_mapping[str(loc_pred)]}, Larger-size: {size_mapping[str(size_pred)]}, Gender: {gender_mapping[str(gender_pred)]}, Age Group: {age_group_mapping[str(age_pred)]}")
            


            # print("Predicted Class:", class_mapping[str(class_pred)])
            # print("Predicted Localization:", localization_mapping[str(loc_pred)])
            # print("Predicted Size:", size_mapping[str(size_pred)])
            # print("Predicted Gender:", gender_mapping[str(gender_pred)])
            # print("Predicted Age Group:", age_group_mapping[str(age_pred)])
         



if __name__ == '__main__':

    # testSingle_image()
    testImage_Folder()

