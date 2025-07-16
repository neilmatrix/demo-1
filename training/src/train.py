import argparse
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 28x28 → 14x14
            nn.Conv2d(16, 32, 3, padding=1), # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 14x14 → 7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def train_model(lr, epochs, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mlflow.set_experiment("mnist-classifier")
    with mlflow.start_run():
        mlflow.log_params({"lr": lr, "epochs": epochs, "batch_size": batch_size})

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        accuracy = correct / total
        mlflow.log_metric("val_accuracy", accuracy)

        # Log model with input example
        sample = torch.randn(1, 1, 28, 28).numpy()
        mlflow.pytorch.log_model(model, name="model", input_example=sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    train_model(args.lr, args.epochs, args.batch_size)