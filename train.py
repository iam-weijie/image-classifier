import argparse
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
import os


def load_model(arch, hidden_units):
    """Load the model."""
    if arch == "resnet18":
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(in_features, hidden_units)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout(0.5)),
                    ("fc2", nn.Linear(hidden_units, 102)),
                    ("output", nn.LogSoftmax(dim=1)),
                ]
            )
        )
    else:
        raise ValueError("Architecture not supported")
    return model


def train(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    """Train the model."""
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    # Data transformations and loaders
    train_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    valid_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    # Load datasets
    train_data = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transforms
    )
    valid_data = datasets.ImageFolder(
        os.path.join(data_dir, "valid"), transform=valid_transforms
    )

    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=64)

    # Get model
    model = load_model(arch, hidden_units)
    model.to(device)

    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # Training loop
    steps = 0
    running_loss = 0
    print_every = 50

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}"
                )

                running_loss = 0
                model.train()

    # Save checkpoint
    checkpoint = {
        "arch": arch,
        "epoch": epochs,
        "state_dict": model.state_dict(),
        "class_to_idx": train_data.class_to_idx,
        "hidden_units": hidden_units,
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
    print(f"Checkpoint saved to {save_dir}/checkpoint.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument("data_dir", help="Directory containing the data")
    parser.add_argument("--save_dir", help="Directory to save checkpoints", default=".")
    parser.add_argument("--arch", help="Architecture of the model", default="resnet18")
    parser.add_argument(
        "--learning_rate", help="Learning rate", type=float, default=0.003
    )
    parser.add_argument(
        "--hidden_units", help="Number of hidden units", type=int, default=250
    )
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=3)
    parser.add_argument("--gpu", help="Use GPU for training", action="store_true")
    args = parser.parse_args()

    train(
        args.data_dir,
        args.save_dir,
        args.arch,
        args.learning_rate,
        args.hidden_units,
        args.epochs,
        args.gpu,
    )
