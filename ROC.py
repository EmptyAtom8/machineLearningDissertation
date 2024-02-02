import os
import torch
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc
import numpy as np
from AD1 import Autoencoder, criterion

ep = 1e-6
desired_size = (28, 28)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = Autoencoder()
model.load_state_dict(torch.load('trailFive.pth', map_location='cpu'))
model.eval()
model.to(device)

# Load and preprocess the test dataset for trash class
test_dataset_trash = datasets.ImageFolder(
    root=r'C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\assets\student_db\student_db\test\1_yes',
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(desired_size),
        transforms.CenterCrop(desired_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
)
test_dataset_trash.class_to_idx = {'1': 1}
folder_path_trash = r"C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\assets\student_db\student_db\test\1_yes"
trash_files = os.listdir(folder_path_trash)
labels_trash = [1] * len(trash_files)
test_dataloader_trash = torch.utils.data.DataLoader(test_dataset_trash, batch_size=32, shuffle=False)

# Calculate anomaly scores for trash class
trash_error = []
for images, _ in test_dataloader_trash:
    images = images.to(device)
    images = images.view(-1, 28 * 28)
    reconstructions = model(images)
    reconstruction_error = (reconstructions - images) ** 2
    anomaly_scores = reconstruction_error.mean(dim=1)
    trash_error.append(anomaly_scores)
trash_error = torch.cat(trash_error)

# Load and preprocess the test dataset for non-trash class
test_dataset_noTrash = datasets.ImageFolder(
    root=r'C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\assets\student_db\student_db\test\0_none',
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(desired_size),
        transforms.CenterCrop(desired_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
)
test_dataset_trash.class_to_idx = {'1': 1}
folder_path_noTrash = r"C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\assets\student_db\student_db\test\0_none"
noTrash_files = os.listdir(folder_path_noTrash)
labels_noTrash = [0] * len(noTrash_files)
test_dataloader_noTrash = torch.utils.data.DataLoader(test_dataset_noTrash, batch_size=32, shuffle=False)

# Calculate anomaly scores for non-trash class
noTrash_error = []
for images, _ in test_dataloader_noTrash:
    images = images.to(device)
    images = images.view(-1, 28 * 28)
    reconstructions = model(images)
    reconstruction_error = (reconstructions - images) ** 2
    anomaly_scores = reconstruction_error.mean(dim=1)
    noTrash_error.append(anomaly_scores)
noTrash_error = torch.cat(noTrash_error)

# Combine the anomaly scores and labels
errors = torch.cat([noTrash_error, trash_error])
labels = torch.cat([torch.zeros(len(noTrash_error)), torch.ones(len(trash_error))])

# Calculate roc and thresholds
fpr, tpr, thresholds = roc_curve(labels.cpu().detach().numpy(), errors.cpu().detach().numpy())

# Calculate AUCPR
auc_roc = auc(fpr, tpr)
print(f'AUC ROC : {auc_roc}')
# Find optimal threshold based on F1 score

optimal_idx = np.argmax(tpr - fpr)
threshold = thresholds[optimal_idx]
print(f"Optimal Threshold based on ROC = {threshold}")

# Evaluate performance on the trash class
total_correct_trash = 0
total_images_trash = 0

for images, labels in test_dataloader_trash:
    images = images.to(device)
    labels = labels.to(device)

    images = images.view(-1, 28 * 28)
    reconstructions = model(images)
    reconstruction_error = (reconstructions - images) ** 2
    anomaly_scores = reconstruction_error.mean(dim=1)

    # Classify images as trash or notrash based on the anomaly score threshold
    predictions = (anomaly_scores > threshold).long()

    # Count the number of correct predictions
    total_correct_trash += (predictions == labels).sum().item()
    total_images_trash += labels.size(0)

accuracy_Trash = total_correct_trash / total_images_trash
truePositive = total_correct_trash
falseNegative = total_images_trash - total_correct_trash

# Evaluate performance on the non-trash class
total_correct_noTrash = 0
total_images_noTrash = 0
for images, labels in test_dataloader_noTrash:
    images = images.to(device)
    labels = labels.to(device)

    images = images.view(-1, 28 * 28)
    reconstructions = model(images)
    reconstruction_error = (reconstructions - images) ** 2
    anomaly_scores = reconstruction_error.mean(dim=1)

    predictions = (anomaly_scores <= threshold).long()

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

print(test_dataset_noTrash.class_to_idx)
print(test_dataset_trash.class_to_idx)
