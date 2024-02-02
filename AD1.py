import torch
import torch.nn as nn
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 12),
                                     nn.ReLU(),
                                     nn.Linear(12, 3)
                                     )
        self.decoder = nn.Sequential(nn.Linear(3, 12),
                                     nn.ReLU(),
                                     nn.Linear(12, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 28 * 28),
                                     nn.Tanh()
                                     )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

if __name__ == "__main__":
    loss_values = []
    desired_size = (28, 28)
    transform = transforms.Compose([
        transforms.Resize(desired_size),
        transforms.CenterCrop(desired_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    nonTrash_dataset = datasets.ImageFolder(
        root=r'C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\assets\student_db\student_db\train\Good',
        transform=transform)
    nonTrash_dataloader = torch.utils.data.DataLoader(nonTrash_dataset, batch_size=32, shuffle=True)
    dataiter = iter(nonTrash_dataloader)
    images, labels = next(dataiter)
    print(torch.min(images), torch.max(images))
    num_epochs = 20
    total_iterations = len(nonTrash_dataloader) * num_epochs
    start_time = time.time()
    outputs = []
    for epoch in range(num_epochs):
        for i, (img, _) in enumerate(nonTrash_dataloader):
            img = img.reshape(-1, 28 * 28).to(device)
            recon = model(img)
            loss = criterion(recon, img.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        current_iteration = epoch * len(nonTrash_dataloader) + i + 1

        progress = current_iteration / total_iterations * 100
        epoch_time = time.time() - start_time
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)
        print(f"Epoch {epoch + 1} took {minutes} minutes {seconds} seconds.")
        print(f"Epoch: {epoch + 1}/{num_epochs}, Iteration: {i + 1}/{len(nonTrash_dataloader)}, "
              f"Loss: {loss.item():,.4f}, Progress: {progress:.2f}%")
        outputs.append((epoch, img, recon))
        loss_values.append(loss.item())

    total_time = time.time() - start_time
    total_minutes = int(total_time // 60)
    total_seconds = int(total_time % 60)
    print(f"Total time: {total_minutes} minutes {total_seconds} seconds.")


    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


    FILE = 'trailFour.pth'

    torch.save(model.state_dict(), FILE)

