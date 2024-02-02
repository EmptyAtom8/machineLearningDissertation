import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset, random_split
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc, roc_curve, f1_score, precision_score
import time


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvolutionalAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
criterion = nn.MSELoss()


def test_model(test_dataloader, threshold):
    TP = 0  # true positive
    TN = 0  # true negative
    FP = 0  # false positive
    FN = 0  # false negative
    scores = []
    ground_truths = []

    with torch.no_grad():
        for img, labels in test_dataloader:
            img = img.to(device)
            recon = model(img)
            errors = ((recon - img) ** 2).view(img.size(0), -1).mean(dim=1).cpu().numpy()
            scores.extend(errors)
            ground_truths.extend(labels.numpy())

            for error, label in zip(errors, labels):
                if error > threshold and label == 1:  # trash
                    TP += 1
                elif error <= threshold and label == 0:  # notrash
                    TN += 1
                elif error > threshold and label == 0:
                    FP += 1
                else:
                    FN += 1

    balanced_accuracy = 0.5 * (TP / (TP + FN)) + 0.5 * (TN / (TN + FP))

    # Compute additional metrics
    precision, recall, _ = precision_recall_curve(ground_truths, scores)
    f1 = f1_score(ground_truths, [1 if s > threshold else 0 for s in scores])
    pr_auc = auc(recall, precision)
    precision_value = precision_score(ground_truths, [1 if s > threshold else 0 for s in scores])

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(ground_truths, scores)
    roc_auc = auc(fpr, tpr)

    # Plotting PR curve
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    # Plotting ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Print metrics
    print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision_value:.4f}')
    print(f'AUC of PR Curve: {pr_auc:.4f}')
    print(f'AUC of ROC Curve: {roc_auc:.4f}')


if __name__ == "__main__":
    loss_values = []
    val_losses = []
    min_val_loss = float('inf')
    best_model_file = 'CAD1_best_model.pth'

    desired_size = (32, 32)
    transform = transforms.Compose([
        transforms.Resize(desired_size),
        transforms.CenterCrop(desired_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(
        root=r'assets/student_db/student_db/train/Good',
        transform=transform
    )

    # set the proportion we want the data be split into
    val_split = 0.2
    num_val_samples = int(len(dataset) * val_split)
    num_train_samples = len(dataset) - num_val_samples

    # set train and validation data set
    train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_epochs = 30
    total_iterations = len(train_dataloader) * num_epochs
    start_time = time.time()
    outputs = []

    for epoch in range(num_epochs):
        model.train()
        for i, (img, _) in enumerate(train_dataloader):
            img = img.to(device)
            recon = model(img)
            loss = criterion(recon, img.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_values.append(loss.item())

        model.eval()
        individual_errors = []  # List to store individual reconstruction errors
        val_loss_epoch = 0
        with torch.no_grad():
            for img, _ in val_dataloader:
                img = img.to(device)
                recon = model(img)
                errors = ((recon - img) ** 2).view(img.size(0), -1).mean(dim=1)  # Compute error for each image in batch
                individual_errors.extend(errors.tolist())  # Store individual errors
                val_loss_epoch += errors.sum().item()

        val_loss_epoch /= len(val_dataset)
        val_losses.append(val_loss_epoch)
        # After computing val_loss_epoch (as in your code)
        # Check if this epoch has a lower validation loss than the minimum seen so far
        if val_loss_epoch < min_val_loss:
            # Update the minimum validation loss
            min_val_loss = val_loss_epoch
            # Save the model weights
            torch.save(model.state_dict(), best_model_file)
            print(f"Validation Loss Improved, Saving Model...")

        epoch_time = time.time() - start_time
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)
        print(f"Epoch {epoch + 1} took {minutes} minutes {seconds} seconds.")
        print(
            f"Epoch: {epoch + 1}/{num_epochs}, Training Loss: {loss.item():,.4f}, Validation Loss: {val_loss_epoch:.4f}")
        # Displaying the maximum error
        max_error = max(individual_errors)
        print(f"Maximum Reconstruction Error: {max_error}")
    chosen_threshold = 0.5
    # Load the test dataset
    test_dataset = datasets.ImageFolder(
        root=r'assets/student_db/student_db/test',
        # This should be the directory containing the 'trash' and 'notrash' subdirectories
        transform=transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Load the best model weights
    model.load_state_dict(torch.load(best_model_file))
    model = model.to(device)
    model.eval()
    # Test the model
    test_model(test_dataloader, chosen_threshold)

    # Plot training and validation losses
    plt.plot(loss_values, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss CAD1')
    plt.legend()
    plt.show()
