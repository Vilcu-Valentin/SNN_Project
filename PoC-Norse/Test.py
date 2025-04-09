import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import norse.torch as norse

# Funcție pentru codificarea Poisson
def poisson_encode(x, time_steps):
    """
    Codifică un batch de imagini, deja aplatizate, într-o secvență de spike-uri.
    x: tensor de forma (batch_size, input_dim) cu valori în [0, 1]
    time_steps: numărul de pași de simulare
    Returnează: tensor de forma (time_steps, batch_size, input_dim)
    """
    return (torch.rand(time_steps, *x.shape, device=x.device) < x).float()

# MNIST data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,))
])
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class NorseNet(nn.Module):
    def __init__(self):
        super(NorseNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.lif1 = norse.LIFCell()
        self.fc2 = nn.Linear(256, 10)
        self.lif2 = norse.LIFCell()

    def forward(self, x):
        z = self.fc1(x)
        spk1, mem1 = self.lif1(z)
        z = self.fc2(spk1)
        spk2, mem2 = self.lif2(z)
        return spk2

# Setup
device = torch.device("cpu")
model = NorseNet().to(device)
model.load_state_dict(torch.load("norse_snn_model.pth"))
model.eval()

# numărul de pași de simulare
sim_steps = 150

# Testarea modelului
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        sys.stdout.write(f"\rProgress: {100 * batch_idx / len(test_loader):.2f}%")
        sys.stdout.flush()

        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        # Codifică imaginile în spike-uri pentru testare
        encoded_images = poisson_encode(images, sim_steps)
        outputs = torch.zeros((images.size(0), 10), device=device)
        for t in range(sim_steps):
            outputs += model(encoded_images[t])

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nTest Accuracy: {100.0 * correct / total:.2f}%")