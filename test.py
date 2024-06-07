"""
This script perform test using the trained model.

Here since the model is trained with only the input image to classify different classes (3 classes).
So for inference the input is an image and the output is the predicted class.

"""




import os
import sys
import torch
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np

from MedMamba import VSSM as medmamba  # import model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # Define data transformations for test images (same as validation)
    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load test dataset
    # test_dataset = datasets.ImageFolder(root="images/split_data/test", transform=data_transform)
    test_dataset = datasets.ImageFolder(root="images/test", transform=data_transform)
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=32, shuffle=False)  # No shuffling for testing

    print("using {} images for testing.".format(test_num))

    # Load the saved model
    num_classes = 3  # Assuming the same number of classes from training
    # model_name = "AmitNew"  # Replace with your model name
    # model_path = "AmitNewNet_trained_2.pth"  # Path to the saved model
    # model_path = "AmitNewNet_trained_3.pth"  # Path to the saved model
    model_path = "AmitNewNet_good.pth"  # Path to the saved model

    net = medmamba(num_classes=num_classes)
    net.load_state_dict(torch.load(model_path))
    net.to(device)

    # Test loop (no training involved here)
    net.eval()  # Set model to evaluation mode
    test_acc = 0.0  # accumulate accurate number / test set
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for data in test_bar:
            images, labels = data
            outputs = net(images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            test_acc += torch.eq(predict_y, labels.to(device)).sum().item()

            all_predictions.extend(predict_y.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    test_accurate = test_acc / test_num
    print('Test Accuracy: %.3f' % test_accurate)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_predictions)
    print("\nConfusion Matrix:")
    print(cm)

    

    # Classification Report
    class_names = ["Without Dysplasia", "OSCC", "With Dysplasia"]  # Replace with your actual class names
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))

    # ... rest of the code for plotting the confusion matrix (optional)



    # Plot Confusion Matrix (using matplotlib)
    import matplotlib.pyplot as plt


    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    # Loop over data and create text annotations.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = cm[i, j]
            props = dict(color="white", fontsize=10)
            ax.text(j, i, text, ha="center", va="center", **props)

    fig.colorbar(im)
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

