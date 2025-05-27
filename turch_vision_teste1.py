import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random


# Transformação de imagem para tensor normalizado
transform = transforms.Compose([
    transforms.ToTensor(),                # Converte imagem [0,255] -> [0,1]
    transforms.Normalize((0.5,), (0.5,))  # Normaliza para -1 a 1
])

# Carregar dataset MNIST
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# Definindo a CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # [1,28,28] → [32,14,14]
        x = self.pool(self.relu(self.conv2(x)))  # [32,14,14] → [64,7,7]
        x = x.view(-1, 64 * 7 * 7)               # achatamento
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Inicializar rede, função de custo e otimizador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento
for epoch in range(5):  # 5 épocas tá bom pra ver resultado inicial
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Época {epoch+1}, Loss: {running_loss / len(train_loader):.4f}')

print("Treinamento completo!")

# Teste com imagens reais
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Acurácia no teste: {100 * correct / total:.2f}%')

test_model()

# Escolher imagem aleatória do test set
image, label = test_set[random.randint(0, len(test_set)-1)]
model.eval()
with torch.no_grad():
    output = model(image.unsqueeze(0).to(device))
    predicted = torch.argmax(output).item()

# Mostrar a imagem
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Label real: {label}, Predito: {predicted}')
plt.show()
