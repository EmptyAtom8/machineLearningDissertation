import os
import torch
import torch.nn as nn
import time
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from AD1 import Autoencoder, criterion

desired_size = (28, 28)
#  set device to gpu priority
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#  load the autoencoder model
model = Autoencoder()
model.load_state_dict(torch.load('trailFive.pth', map_location='cpu'))
model.eval()
model.to(device)
#  load test images for trash data set
test_dataset_trash = datasets.ImageFolder(
    root=r'C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\assets\student_db\student_db\test\1_yes',
    transform=transforms.Compose([
        transforms.Resize(desired_size),
        transforms.CenterCrop(desired_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
)

folder_path_trash = r"C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\assets\student_db\student_db\test\1_yes"
trash_files = os.listdir(folder_path_trash)
labels = [1] * len(trash_files)
test_dataloader_trash = torch.utils.data.DataLoader(test_dataset_trash, batch_size=32, shuffle=False)
threshold = 0.5


total_correct_trash = 0
total_images_trash = 0

for images, labels in test_dataloader_trash:
    images = images.to(device)
    labels = labels.to(device)

    images = images.view(-1, 28 * 28)
    reconstructions = model(images)
    loss = criterion(reconstructions, images)
    anomaly_scores = loss.mean(dim=0)  # Calculate the mean loss for each image

    # Classify images as trash or notrash based on the anomaly score threshold
    predictions = anomaly_scores > threshold

    # Count the number of correct predictions
    total_correct_trash += (predictions == labels).sum().item()
    total_images_trash += labels.size(0)

accuracy_Trash = total_correct_trash / total_images_trash
truePositive = total_correct_trash
falseNegative = total_images_trash - total_correct_trash
#  testing non-trash images
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
# Get the list of file names in the folder
noTrash_files = os.listdir(folder_path_noTrash)
labels = [0] * len(noTrash_files)
test_dataloader_noTrash = torch.utils.data.DataLoader(test_dataset_noTrash, batch_size=32, shuffle=False)

total_correct_noTrash = 0
total_images_noTrash = 0
for images, labels in test_dataloader_noTrash:
    images = images.to(device)
    labels = labels.to(device)

    images = images.view(-1, 28 * 28)
    reconstructions = model(images)
    loss = criterion(reconstructions, images)
    anomaly_scores = loss.mean(dim=0)

    predictions = anomaly_scores < threshold

    total_correct_noTrash += (predictions == labels).sum().item()
    total_images_noTrash += labels.size(0)

accuracy_nonTrash = total_correct_noTrash / total_images_noTrash
trueNegative = total_correct_noTrash
falsePositive = total_images_noTrash - total_correct_noTrash
BalancedAccuracy = 0.5 * (truePositive / (truePositive + falseNegative)) + 0.5 * (
            trueNegative / (trueNegative + falsePositive))
print(f"BalanceAccuracy = {BalancedAccuracy}")
print(f't Value = {threshold}')
recall = truePositive/ (truePositive+falseNegative)
print(f'Recall Value: {recall}')
precision = truePositive / (truePositive + falsePositive)
print(f'Precision: {precision}')
f1 = 2*(precision*recall)/(precision+recall)
print(f'F1 Score: {f1}')

