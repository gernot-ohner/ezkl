
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.layer(x))

net = SimpleNet()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=5)

inputs = torch.tensor([[0.], [1.]], dtype=torch.float32)
targets = torch.tensor([[0.], [1.]], dtype=torch.float32)

epochs = 1000

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}')

print("\nTraining complete.")
net.eval()

with torch.no_grad():
    test_input = torch.tensor([[0.], [1.]], dtype=torch.float32)
    test_output = net(test_input)
    print(f"Test Input: {test_input.view(-1).numpy()}, Test Output: {test_output.view(-1).numpy()}")

def write_onnx():
    onnx_file_path = "/Users/gernotohner/dev/rust/ezkl/examples/0l_simplest_network/onnx/simple_net.onnx"
    torch.onnx.export(net, dummy_input, onnx_file_path, export_params=True, opset_version=10,
                     do_constant_folding=True, input_names=['input'], output_names=['output'])

    print(onnx_file_path)

def write_dummy_input():
    dummy_input = torch.tensor([[0.]], dtype=torch.float32)

    input_list = dummy_input.tolist()
    with open('input.json', 'w') as json_file:
        json.dump(input_list, json_file)


# ----------------------------------------------------------
# Running network and measuring inference time
# ----------------------------------------------------------

# Number of runs for measuring inference time
num_runs = 1000
times = []

with torch.no_grad():
    for _ in range(num_runs):
        start_time = time.time_ns()
        _ = net(test_input)
        end_time = time.time_ns()

        # Calculate and store the time taken for this run
        times.append(end_time - start_time)

# Calculate and print statistics
times = np.array(times)
mean_time = np.mean(times)
std_dev = np.std(times)

print(f"Mean inference time: {mean_time:.2f} ns")
print(f"Standard deviation: {std_dev:.2f} ns")


