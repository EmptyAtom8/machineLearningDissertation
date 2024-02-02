import os
import torch
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_curve, auc
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

# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(labels.cpu().detach().numpy(), errors.cpu().detach().numpy())

# Calculate AUCPR
aucpr = auc(recall, precision)
print(f'AUCPR : {aucpr}')
# Find optimal threshold based on F1 score
f1_scores = 2 * (precision * recall) / (precision + recall + (1e-6))
threshold_index = np.nanargmax(f1_scores)
threshold = thresholds[threshold_index]

# Evaluate performance on the trash class
total_correct_trash = 0
total_images_trash = 0
true_labels_trash = torch.ones((32, ), dtype=torch.long).to(device)  # Create labels tensor (1 for trash)

for images, _ in test_dataloader_trash:  # Here I'm using _ instead of labels
    images = images.to(device)
    images = images.view(images.shape[0], -1)

    reconstructions = model(images)
    reconstruction_error = (reconstructions - images) ** 2
    anomaly_scores = reconstruction_error.mean(dim=1)

    # Classify images as trash or notrash based on the anomaly score threshold
    predictions = (anomaly_scores > threshold).long()
    true_labels_trash = torch.ones(len(predictions), dtype=torch.long).to(device)
    # Count the number of correct predictions
    total_correct_trash += (predictions == true_labels_trash[:len(predictions)]).sum().item()
    total_images_trash += len(predictions)

accuracy_Trash = total_correct_trash / total_images_trash
truePositive = total_correct_trash
falseNegative = total_images_trash - total_correct_trash

# Evaluate performance on the non-trash class
total_correct_noTrash = 0
total_images_noTrash = 0
true_labels_noTrash = torch.zeros((32, ), dtype=torch.long).to(device)  # Create labels tensor (0 for non-trash)

for images, _ in test_dataloader_noTrash:  # Here I'm using _ instead of labels
    images = images.to(device)

    images = images.view(images.shape[0], -1)
    reconstructions = model(images)
    reconstruction_error = (reconstructions - images) ** 2
    anomaly_scores = reconstruction_error.mean(dim=1)

    predictions = (anomaly_scores <= threshold).long()
    true_labels_noTrash = torch.zeros(len(predictions), dtype=torch.long).to(device)

    total_correct_noTrash += (predictions == true_labels_noTrash[:len(predictions)]).sum().item()
    total_images_noTrash += len(predictions)

accuracy_nonTrash = total_correct_noTrash / total_images_noTrash
trueNegative = total_correct_noTrash
falsePositive = total_images_noTrash - total_correct_noTrash
BalancedAccuracy = 0.5 * (truePositive / (truePositive + falseNegative)) + 0.5 * (
            trueNegative / (trueNegative + falsePositive))
print(f"BalanceAccuracy = {BalancedAccuracy}")
print(f't Value = {threshold}')
recall = truePositive / (truePositive+falseNegative)
print(f'Recall Value: {recall}')
precision = truePositive / (truePositive + falsePositive)
print(f'Precision: {precision}')
f1 = 2*(precision*recall)/(precision+recall)
print(f'F1 Score: {f1}')

print(test_dataset_noTrash.class_to_idx)
print(test_dataset_trash.class_to_idx)
