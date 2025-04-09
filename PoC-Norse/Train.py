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
    time_steps: numărul de pași de simulare (fiecare pas va produce un set de spike-uri)
    Returnează: tensor de forma (time_steps, batch_size, input_dim)
    """
    return (torch.rand(time_steps, *x.shape, device=x.device) < x).float()

# MNIST data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,))
])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class NorseNet(nn.Module):
    def __init__(self):
        super(NorseNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)  # Input to hidden layer
        self.lif1 = norse.LIFCell()        # First spiking layer
        self.fc2 = nn.Linear(256, 10)      # Hidden to output layer
        self.lif2 = norse.LIFCell()        # Second spiking layer

    def forward(self, x):
        z = self.fc1(x)
        spk1, mem1 = self.lif1(z)
        z = self.fc2(spk1)
        spk2, mem2 = self.lif2(z)
        return spk2  # Returnează doar spike-urile finale

# Training setup
device = torch.device("cpu")
model = NorseNet().to(device)

# Întreabăm utilizatorul dacă dorește să încarce modelul salvat anterior
load_model = input("Dorești să încarci modelul salvat anterior? (yes/no): ").strip().lower()
if load_model == "yes":
    try:
        model.load_state_dict(torch.load("norse_snn_model.pth"))
        print("Modelul salvat a fost încărcat cu succes. Continuăm antrenarea.")
    except FileNotFoundError:
        print("Modelul salvat nu a fost găsit. Vom începe antrenarea de la zero.")
else:
    print("Vom începe antrenarea de la zero.")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# numărul de pași de simulare (time steps)
sim_steps = 25

# Training loop
epochs = 75
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    print(f"Epoch {epoch+1}/{epochs}")
    for batch_idx, (images, labels) in enumerate(train_loader):
        sys.stdout.write(f"\rProgress: {100 * batch_idx / len(train_loader):.2f}%")
        sys.stdout.flush()

        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        # Codifică imaginile în spike-uri folosind codificarea Poisson
        encoded_images = poisson_encode(images, sim_steps)
        optimizer.zero_grad()
        outputs = torch.zeros((images.size(0), 10), device=device)
        for t in range(sim_steps):
            outputs += model(encoded_images[t])
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"\nEpoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Salvăm modelul antrenat
torch.save(model.state_dict(), "norse_snn_model.pth")
print("Modelul actualizat a fost salvat în norse_snn_model.pth.")