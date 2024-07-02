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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd



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
       


def load_data(csv_file):
    df = pd.read_csv(csv_file)
    
    # Encode categorical variables
    df['localization'] = df['localization'].astype('category')
    df['localization'] = df['localization'].cat.codes
    
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})
    
    df['age_group'] = df['age_group'].astype('category')
    df['age_group'] = df['age_group'].cat.codes

    df['larger_size'] = df['larger_size'].astype('category')
    df['larger_size'] = df['larger_size'].cat.codes
    
    df['Muthukumar_Classes'] = df['Muthukumar_Classes'].astype('category')
    df['Muthukumar_Classes'] = df['Muthukumar_Classes'].cat.codes
    
    return df



class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = load_data(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 3])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, -1]
        
        localization = self.img_labels.iloc[idx, 4]
        larger_size = self.img_labels.iloc[idx, 5]
        gender = self.img_labels.iloc[idx, 6]
        age_group = self.img_labels.iloc[idx, 7]
        
        additional_features = torch.tensor([localization, larger_size, gender, age_group], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, additional_features, label




# class CustomImageDataset(torch.utils.data.Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None):
#         self.img_labels = load_data(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 3])
#         image = load_image(img_path, self.transform)
#         label = self.img_labels.iloc[idx, -1]
        
#         localization = self.img_labels.iloc[idx, 4]
#         larger_size = self.img_labels.iloc[idx, 5]
#         gender = self.img_labels.iloc[idx, 6]
#         age_group = self.img_labels.iloc[idx, 7]
        
#         additional_features = torch.tensor([localization, larger_size, gender, age_group], dtype=torch.long)
        
#         return image, additional_features, label



def compute_metrics(model, device, test_loader, num_classes, num_localizations, num_sizes, num_genders, num_age_groups):
    model.eval()
    class_criterion = nn.CrossEntropyLoss()
    loc_criterion = nn.CrossEntropyLoss()
    size_criterion = nn.CrossEntropyLoss()
    gender_criterion = nn.CrossEntropyLoss()
    age_criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct_class = 0
    correct_loc = 0
    correct_size = 0
    correct_gender = 0
    correct_age = 0
    total_samples = 0

    all_class_preds = []
    all_class_labels = []

    with torch.no_grad():
        for images, features, labels in test_loader:
            images = images.to(device)
            features = features.to(device).long()
            labels = labels.to(device).long()
            
            class_outputs, loc_outputs, size_outputs, gender_outputs, age_outputs = model(images)
            
            loss_class = class_criterion(class_outputs, labels)
            loss_loc = loc_criterion(loc_outputs, features[:, 0])
            loss_size = size_criterion(size_outputs, features[:, 1])
            loss_gender = gender_criterion(gender_outputs, features[:, 2])
            loss_age = age_criterion(age_outputs, features[:, 3])
            
            total_loss += (loss_class + loss_loc + loss_size + loss_gender + loss_age).item()
            
            class_preds = class_outputs.argmax(dim=1)
            loc_preds = loc_outputs.argmax(dim=1)
            size_preds = size_outputs.argmax(dim=1)
            gender_preds = gender_outputs.argmax(dim=1)
            age_preds = age_outputs.argmax(dim=1)

            correct_class += (class_preds == labels).sum().item()
            correct_loc += (loc_preds == features[:, 0]).sum().item()
            correct_size += (size_preds == features[:, 1]).sum().item()
            correct_gender += (gender_preds == features[:, 2]).sum().item()
            correct_age += (age_preds == features[:, 3]).sum().item()
            total_samples += labels.size(0)

            all_class_preds.extend(class_preds.cpu().numpy())
            all_class_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    class_accuracy = correct_class / total_samples
    loc_accuracy = correct_loc / total_samples
    size_accuracy = correct_size / total_samples
    gender_accuracy = correct_gender / total_samples
    age_accuracy = correct_age / total_samples

    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Class Accuracy: {class_accuracy:.4f}")
    print(f"Localization Accuracy: {loc_accuracy:.4f}")
    print(f"Size Accuracy: {size_accuracy:.4f}")
    print(f"Gender Accuracy: {gender_accuracy:.4f}")
    print(f"Age Group Accuracy: {age_accuracy:.4f}")

    cm = confusion_matrix(all_class_labels, all_class_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(num_classes)])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


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
  


def testImage_Folder_report():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    data_transform = transforms.Compose([transforms.Resize((224, 224)),
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

    # Prepare test dataset from CSV
    test_dataset = CustomImageDataset(annotations_file="images/dataset_B.csv", img_dir="images/dataset_B", transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    compute_metrics(model, device, test_loader, num_classes, num_localizations, num_sizes, num_genders, num_age_groups)


if __name__ == '__main__':

    testSingle_image()
    # testImage_Folder()
    # testImage_Folder_report()   # See the running output in the file /home/coe_iot_ai/Desktop/Amit/medHisPath/running_output.txt

