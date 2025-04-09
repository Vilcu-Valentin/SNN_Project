import os
import torch
import torch.nn as nn
import norse.torch as norse
from PIL import Image
from numpy import array


class NorseNet(nn.Module):
    def __init__(self):
        super(NorseNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.lif1 = norse.LIFCell()
        self.fc2 = nn.Linear(256, 10)
        self.lif2 = norse.LIFCell()

    def forward(self, x):
        z = self.fc1(x)
        spk1, _ = self.lif1(z)
        z = self.fc2(spk1)
        spk2, _ = self.lif2(z)
        return spk2


def poisson_encode(x, time_steps):
    return (torch.rand(time_steps, *x.shape, device=x.device) < x).float()


device = torch.device("cpu")
model = NorseNet().to(device)
model.load_state_dict(torch.load("norse_snn_model.pth", map_location=device))
model.eval()

test_dir = "./test-formatted"
files = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])

sim_steps = 150

for file in files:
    image_path = os.path.join(test_dir, file)
    image = Image.open(image_path).convert("L")

    # Transformă imaginea într-un array și apoi într-un tensor
    image = torch.tensor(array(image), dtype=torch.float32, device=device).view(1, -1) / 255.0

    encoded_image = poisson_encode(image, sim_steps)

    # Forward pass prin model
    outputs = torch.zeros((image.size(0), 10), device=device)
    with torch.no_grad():
        for t in range(sim_steps):
            outputs += model(encoded_image[t])
        _, predicted = torch.max(outputs, 1)

    print(f"Predicția pentru {file}: {predicted.item()}")
