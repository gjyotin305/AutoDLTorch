import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

# 4806 -> 3280 Free Memory Release practically

# -----------------------------
# Dense Layer (core building block)
# -----------------------------
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = torch.cat([x, out], 1)  # concatenate input and output
        return out


# -----------------------------
# Dense Block
# -----------------------------
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = []
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x
        # return self.block(x)


# -----------------------------
# Transition Layer
# -----------------------------
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = checkpoint(self.conv, out)
        out = self.pool(out)
        # out = self.pool(self.conv(self.relu(self.bn(x))))
        return out


# -----------------------------
# DenseNet Model
# -----------------------------
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=10):
        super(DenseNet, self).__init__()
        num_init_features = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)

        # Dense Blocks + Transitions
        channels = num_init_features
        self.features = nn.Sequential()
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, channels, growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            channels = channels + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionLayer(channels, channels // 2)
                self.features.add_module(f'transition{i+1}', trans)
                channels = channels // 2

        # Final batch norm
        self.bn = nn.BatchNorm2d(channels)

        # Classifier
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        # out = checkpoint_sequential(self.features, 2, x)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# -----------------------------
# Training Script with tqdm
# -----------------------------
def train_model():
    # Data preparation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=10).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        loop = tqdm(trainloader, desc=f"Epoch [{epoch+1}/10]", leave=False)

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=f"{running_loss/len(trainloader):.4f}")

        print(f"Epoch [{epoch+1}/10] - Avg Loss: {running_loss/len(trainloader):.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"\n✅ Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    train_model()