import os
import torch
import torch.nn as nn
from torchvision import transforms
import norse.torch as norse
from PIL import Image
from torchvision.transforms import ToPILImage


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

transform_Custom = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (1 - x).float())
])

sim_steps = 150
test_dir = "./test"
files = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])
to_pil = ToPILImage()

for file in files:
    image_path = os.path.join(test_dir, file)
    image = Image.open(image_path).convert("L")
    image = transform_Custom(image)

    # Debug: afişăm statistici și imaginea preprocesată
    print(f"{file} - min: {image.min().item()}, max: {image.max().item()}, mean: {image.mean().item()}")
    # Pentru a vizualiza imaginea, de-comentează:
    #pil_img = to_pil(image.cpu())
    #pil_img.show()  # sau pil_img.save("debug_" + file)

    image = image.view(1, -1).to(device)
    encoded_image = poisson_encode(image, sim_steps)

    outputs = torch.zeros((image.size(0), 10), device=device)
    with torch.no_grad():
        for t in range(sim_steps):
            outputs += model(encoded_image[t])
        _, predicted = torch.max(outputs, 1)

    print(f"Predicția pentru {file}: {predicted.item()}")