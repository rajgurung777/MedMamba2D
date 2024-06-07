"""
This script perform training using input as the image along with additional features.

Here since the model is trained with the additional features to classify different classes (5 classes).
So accordingly for inference, the input should be image along with the additional features so that the output class is predicted.

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

from MedMamba import VSSM as medmamba  # import model
from PIL import Image

import torch.nn.functional as F



def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df['localization'] = df['localization'].astype('category').cat.codes
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})
    df['age_group'] = df['age_group'].astype('category').cat.codes
    return df

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = load_data(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.img_labels['Muthukumar_Classes'] = self.label_encoder.fit_transform(self.img_labels['Muthukumar_Classes'])

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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = CustomImageDataset(annotations_file="images/dataset_B.csv", 
                                       img_dir="images/dataset_B", 
                                       transform=data_transform["train"])
    train_num = len(train_dataset)
    print("train_num =", train_num)

    flower_list = train_dataset.img_labels['Muthukumar_Classes'].unique().tolist()
    cla_dict = {i: flower_list[i] for i in range(len(flower_list))}
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    num_classes = len(flower_list)
    original_model = medmamba(num_classes=num_classes)
    net = VSSMWithFeatures(original_model, num_additional_features=4, num_classes=num_classes)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 100
    best_acc = 0.0
    save_path = './{}Net.pth'.format('AmitNew')
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
            outputs = net(images.to(device), features.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
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
