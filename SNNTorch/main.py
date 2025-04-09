import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import surrogate

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# data loading
transform = transforms.Compose([
    transforms.ToTensor(), # converts image to tensor (values 0.0-1.0)
    transforms.Normalize((0.0,), (1.0,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Network parameters
num_inputs = 28*28
num_hidden = 256
num_outputs = 10
beta = 0.95 # membrane leak constant for LIF neurons
num_steps = 25 # simulation time steps for each input

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2)
        return spk1, spk2

# Instantiate the network and move to GPU if available
device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)

# Training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

num_epochs = 10
net.train()  # set network in training mode

for epoch in range(num_epochs):
    total_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        images = images.view(images.size(0), -1)

        net.lif1.init_leaky()  # reset hidden layer neurons
        net.lif2.init_leaky()  # reset output layer neurons

        output_spike_counts = torch.zeros((images.size(0), num_outputs), device=device)

        for step in range(num_steps):
            _, spk_out = net(images)
            output_spike_counts += spk_out

        loss = loss_fn(output_spike_counts, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")

# Evaluation
net.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        images = images.view(images.size(0), -1)

        net.lif1.init_leaky()
        net.lif2.init_leaky()
        output_spike_counts = torch.zeros((images.size(0), num_outputs), device=device)

        for step in range(num_steps):
            _, spk_out = net(images)
            output_spike_counts += spk_out
        _, predicted = torch.max(output_spike_counts, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100.0 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
torch.save(net.state_dict(), "snn_model.pth")
