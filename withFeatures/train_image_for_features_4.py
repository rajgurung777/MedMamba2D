"""
Multi-task Learning: Predicts multiple outputs one for each features and class.

Trains a model with multiple heads/labels, each predicting an independent feature.

This is interesting because we had a dataset where the labels are available (CSV file). So we train the model with multiple independent labels.
Now the interesting task we can perform prediction of these multiple labels with only an input image. This enables us to use an unlabelled data (input image) and predicts its likely label.

Reference file: test_image_for_features_4.py
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
import pandas as pd
from MedMamba import VSSM as medmamba
from PIL import Image
import torch.nn.functional as F

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    
    # Encode categorical variables
    df['localization'] = df['localization'].astype('category')
    localization_mapping = dict(enumerate(df['localization'].cat.categories))
    df['localization'] = df['localization'].cat.codes
    
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})
    gender_mapping = {0: 'M', 1: 'F'}
    
    df['age_group'] = df['age_group'].astype('category')
    age_group_mapping = dict(enumerate(df['age_group'].cat.categories))
    df['age_group'] = df['age_group'].cat.codes

    df['larger_size'] = df['larger_size'].astype('category')
    larger_size_mapping = dict(enumerate(df['larger_size'].cat.categories))
    df['larger_size'] = df['larger_size'].cat.codes
    
    df['Muthukumar_Classes'] = df['Muthukumar_Classes'].astype('category')
    class_mapping = dict(enumerate(df['Muthukumar_Classes'].cat.categories))
    df['Muthukumar_Classes'] = df['Muthukumar_Classes'].cat.codes
    
    mappings = {
        'localization': localization_mapping,
        'gender': gender_mapping,
        'age_group': age_group_mapping,
        'larger_size': larger_size_mapping,
        'Muthukumar_Classes': class_mapping
    }
    
    # Save mappings to JSON file
    with open('feature_mappings.json', 'w') as json_file:
        json.dump(mappings, json_file, indent=4)
    
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

    # Load the mappings from JSON file
    with open('feature_mappings.json', 'r') as json_file:
        feature_mappings = json.load(json_file)

    num_classes = len(feature_mappings['Muthukumar_Classes'])
    num_localizations = len(feature_mappings['localization'])
    num_sizes = len(feature_mappings['larger_size'])
    num_genders = len(feature_mappings['gender'])
    num_age_groups = len(feature_mappings['age_group'])

    print("Total classes:", num_classes)
    print("Total localizations:", num_localizations)
    print("Total sizes:", num_sizes)
    print("Total genders:", num_genders)
    print("Total age groups:", num_age_groups)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # Number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    original_model = medmamba(num_classes=num_classes)
    net = VSSMWithMultipleHeads(original_model, num_classes, num_localizations, num_sizes, num_genders, num_age_groups)
    net.to(device)
    
    class_loss_function = nn.CrossEntropyLoss()
    loc_loss_function = nn.CrossEntropyLoss()
    size_loss_function = nn.CrossEntropyLoss()
    gender_loss_function = nn.CrossEntropyLoss()
    age_loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 100
    best_acc = 0.0
    save_path = './{}Net.pth'.format('AmitMultiClass')
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
            labels = labels.long().to(device)  # Ensure labels are of type long
            features = features.long().to(device)  # Ensure features are of type long
            
            optimizer.zero_grad()
            class_outputs, loc_outputs, size_outputs, gender_outputs, age_outputs = net(images.to(device))
            loss_class = class_loss_function(class_outputs, labels)
            loss_loc = loc_loss_function(loc_outputs, features[:, 0])
            loss_size = size_loss_function(size_outputs, features[:, 1])
            loss_gender = gender_loss_function(gender_outputs, features[:, 2])
            loss_age = age_loss_function(age_outputs, features[:, 3])
            
            loss = loss_class + loss_loc + loss_size + loss_gender + loss_age
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(class_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
