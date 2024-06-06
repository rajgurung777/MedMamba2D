import os
import sys
import torch
from torchvision import transforms, datasets
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from MedMamba import VSSM as medmamba  # import model

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load dataset
    dataset = datasets.ImageFolder(root="images/dataset_A", transform=data_transform)
    dataset_size = len(dataset)
    print("Total dataset size: ", dataset_size)

    # Define k-fold cross validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    num_classes = 3
    model_path = "AmitNewNet_good.pth"  # Path to the saved model
    results = {}

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold}')

        # Sample elements randomly from a given list of ids, no replacement.
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loader for testing data in this fold
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, sampler=test_subsampler)

        # Initialize model
        net = medmamba(num_classes=num_classes)
        net.load_state_dict(torch.load(model_path))
        net.to(device)

        # Evaluation for this fold
        net.eval()
        test_acc = 0.0
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

        test_accurate = test_acc / len(test_ids)
        print(f'Fold {fold} Test Accuracy: {test_accurate}')
        results[fold] = test_accurate

        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_predictions)
        print("\nConfusion Matrix:")
        print(cm)
        plot_confusion_matrix(cm, classes=["Without Dysplasia", "OSCC", "With Dysplasia"], title=f'Confusion Matrix for Fold {fold}')

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum_acc = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value}')
        sum_acc += value
    print(f'Average: {sum_acc/len(results)}')

if __name__ == '__main__':
    main()
