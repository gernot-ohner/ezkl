import time
import torch
import json
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import onnx

print(onnx.__version__)

# Step 1: Data Preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# Step 2: Model Architecture
class SimpleMNISTClassifier(nn.Module):
    def __init__(self):
        super(SimpleMNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = SimpleMNISTClassifier()

# Step 3: Training
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Step 4: Evaluation
correct = 0
total = 0
times = []
with torch.no_grad():
    for data in testloader:
        images, labels = data

        start_time = time.time()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        end_time = time.time()

        times.append(end_time - start_time)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
# Calculate and print statistics
times = np.array(times)
mean_time = np.mean(times)
std_dev = np.std(times)

print(f"Mean inference time: {mean_time * 1000:.2f} ms")
print(f"Standard deviation: {std_dev * 1000:.2f} ms")

def export_onnx():
    # Step 5: Export to ONNX
    dummy_input = torch.randn(64, 1, 28, 28)
    torch.onnx.export(model, dummy_input, "simple_mnist.onnx")


def write_sample_input():
    # Generate synthetic data as a placeholder (replace this with an actual MNIST image in practice)
    # A 28x28 grayscale image would be represented as a 28x28 array.
    sample_input = np.random.rand(1, 28, 28).tolist()

    # Serialize the NumPy array to JSON formatted data
    with open("input.json", "w") as f:
        json.dump(sample_input, f)

    # Check if the file was created and its content
    with open("input.json", "r") as f:
        loaded_input = json.load(f)

    loaded_input[:1]  # Display a small portion to check
