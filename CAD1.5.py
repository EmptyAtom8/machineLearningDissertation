import time

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split


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
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # Changed this layer
            nn.Sigmoid()  # to get pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvolutionalAutoencoder().to(device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
criterion = nn.MSELoss()

#######################################################


if __name__ == "__main__":


    #  pre-processing images
    desired_size = (32, 32)
    transform = transforms.Compose([
        #  transforms.Grayscale(num_output_channels=1),
        transforms.Resize(desired_size),
        transforms.CenterCrop(desired_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # combine all three types of images into one data set
    dataset1 = datasets.ImageFolder(
        root=r'C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\assets\student_db\student_db\train\Good',
        transform=transform)
    #dataset2 = datasets.ImageFolder(
         #root=r'C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\assets\student_db\student_db\train\MayBe',
        #transform=transform)
    dataset3 = datasets.ImageFolder(
        root=r'C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\assets\student_db\student_db\train\NoGood',
        transform=transform)
    combined_dataset = ConcatDataset([dataset1, dataset3])

    # set the proportion we want the data be split into
    val_split = 0.2
    num_val_samples = int(len(combined_dataset) * val_split)
    num_train_samples = len(combined_dataset) - num_val_samples

    # set train and validation data set
    train_dataset, val_dataset = random_split(combined_dataset, [num_train_samples, num_val_samples])

    dataiter = iter(train_dataset)
    images, labels = next(dataiter)
    print(torch.min(images), torch.max(images))

    # set training and validation data loader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_epochs = 20
    total_iterations = len(train_dataloader) * num_epochs
    start_time = time.time()
    outputs = []
    val_losses = []
    loss_values = []
    for epoch in range(num_epochs):
        model.train()
        for i, (img, _) in enumerate(train_dataloader):
            img = img.to(device)
            recon = model(img)
            loss = criterion(recon, img.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        current_iteration = epoch * len(train_dataloader) + i + 1

        progress = current_iteration / total_iterations * 100
        epoch_time = time.time() - start_time
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)
        print(f"Epoch {epoch + 1} took {minutes} minutes {seconds} seconds.")
        print(f"Epoch: {epoch + 1}/{num_epochs}, Iteration: {i + 1}/{len(train_dataloader)}, "
              f"Loss: {loss.item():,.4f}, Progress: {progress:.2f}%")
        outputs.append((epoch, img, recon))
        loss_values.append(loss.item())

        model.eval()
        val_losses_epoch = []

        with torch.no_grad():
            for img, _ in val_dataloader:
                img = img.to(device)
                recon = model(img)
                loss = criterion(recon, img)
                val_losses_epoch.append(loss.item())

        # Calculate average losses
        avg_train_loss = sum(loss_values) / len(loss_values)
        avg_val_loss = sum(val_losses_epoch) / len(val_losses_epoch)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

    total_time = time.time() - start_time
    total_minutes = int(total_time // 60)
    total_seconds = int(total_time % 60)
    print(f"Total time: {total_minutes} minutes {total_seconds} seconds.")

    plt.plot(loss_values, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss CAD1.5 with Trash and Non Trash')
    plt.legend()
    plt.show()
    FILE = 'trailSeven.pth'

    torch.save(model.state_dict(), FILE)
