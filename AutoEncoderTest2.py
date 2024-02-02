import os
import numpy as np
import torch
import torch.nn as nn
import time
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from AD1 import Autoencoder, criterion

desired_size = (28, 28)

# Set device to GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the autoencoder model
model = Autoencoder()
model.load_state_dict(torch.load('trailFour.pth', map_location='cpu'))
model.eval()
model.to(device)
# Load test images for trash data set
test_dataset_trash = datasets.ImageFolder(
    root=r'C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\assets\student_db\student_db\test\1_yes',
    transform=transforms.Compose([
        transforms.Resize(desired_size),
        transforms.CenterCrop(desired_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
)
# Label the images in the folder as 1 to indicate trash
folder_path_trash = r"C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\assets\student_db\student_db\test\1_yes"
trash_files = os.listdir(folder_path_trash)
labels_trash = [1] * len(trash_files)
test_dataset_trash.targets = labels_trash

# Testing non-trash images
test_dataset_noTrash = datasets.ImageFolder(
    root=r'C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\assets\student_db\student_db\test\0_none',
    transform=transforms.Compose([
        transforms.Resize(desired_size),
        transforms.CenterCrop(desired_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
)

folder_path_noTrash = r"C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\assets\student_db\student_db\test\0_none"
noTrash_files = os.listdir(folder_path_noTrash)
print ('non trash loading complete')
labels_noTrash = [0] * len(noTrash_files)
test_dataset_noTrash.targets = labels_noTrash

# Set up data loaders
test_dataloader_trash = torch.utils.data.DataLoader(test_dataset_trash, batch_size=32, shuffle=False)
test_dataloader_noTrash = torch.utils.data.DataLoader(test_dataset_noTrash, batch_size=32, shuffle=False)

thresholds = np.arange(0.02, 0.08, 0.001)  # Generate threshold values from 0.2 to 1
print('trash loading complete')
balanced_accuracies = []
precisions = []
f1_scores = []

# Set the model to evaluation mode
model.eval()

for threshold in thresholds:
    total_correct_trash = 0
    total_images_trash = 0

    for images, labels_trash in test_dataloader_trash:
        images = images.to(device)
        labels_trash = labels_trash.to(device)

        images = images.view(-1, 28 * 28)
        reconstructions = model(images)
        loss = criterion(reconstructions, images)
        anomaly_scores = loss.mean(dim=0)

        # Classify images as trash or non-trash based on the threshold
        predictions = anomaly_scores > threshold

        # Count the number of correct predictions
        total_correct_trash += (predictions == labels_trash).sum().item()
        total_images_trash += labels_trash.size(0)

    accuracy_trash = total_correct_trash / total_images_trash
    truePositive = total_correct_trash
    falseNegative = total_images_trash - total_correct_trash
    print(f't-value: {threshold}')


    total_correct_noTrash = 0
    total_images_noTrash = 0

    for images, labels_noTrash in test_dataloader_noTrash:
        images = images.to(device)
        labels_noTrash = labels_noTrash.to(device)

        images = images.view(-1, 28 * 28)
        reconstructions = model(images)
        loss = criterion(reconstructions, images)
        anomaly_scores = loss.mean(dim=0)

        # Classify images as trash or non-trash based on the threshold
        predictions = anomaly_scores < threshold

        # Count the number of correct predictions
        total_correct_noTrash += (predictions == labels_noTrash).sum().item()
        total_images_noTrash += labels_noTrash.size(0)

    accuracy_noTrash = total_correct_noTrash / total_images_noTrash
    trueNegative = total_correct_noTrash
    falsePositive = total_images_noTrash - total_correct_noTrash

    balanced_accuracy = 0.5 * (truePositive / (truePositive + falseNegative)) + 0.5 * (trueNegative / (trueNegative + falsePositive))
    balanced_accuracies.append(balanced_accuracy)

    recall = truePositive / (truePositive + falseNegative)
    precision = truePositive / (truePositive + falsePositive)
    f1 = 2 * (precision * recall) / (precision + recall)




    precisions.append(precision)
    f1_scores.append(f1)

plt.plot(thresholds, balanced_accuracies)
plt.xlabel('Threshold (t-value)')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy vs Threshold')
plt.show()

plt.plot(thresholds, precisions)
plt.xlabel('Threshold (t-value)')
plt.ylabel('Precision')
plt.title('Precision vs Threshold')
plt.show()

plt.plot(thresholds, f1_scores)
plt.xlabel('Threshold (t-value)')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Threshold')
plt.show()
