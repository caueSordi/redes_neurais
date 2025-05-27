import torch
import torch.nn as nn
import torch.optim as optim

# Dados de entrada (XOR)
inputs = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=torch.float32)

# Saídas esperadas
targets = torch.tensor([
    [0],
    [1],
    [1],
    [0]
], dtype=torch.float32)

# Definindo a arquitetura da rede
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(2, 2)   # camada oculta com 2 neurônios
        self.output = nn.Linear(2, 1)   # camada de saída com 1 neurônio
        self.sigmoid = nn.Sigmoid()    # ativação

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Instanciar modelo
model = XORNet()

# Função de perda e otimizador
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Treinamento
for epoch in range(10000):
    # Forward
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log a cada 1000 épocas
    if epoch % 1000 == 0:
        print(f'Época {epoch}, Loss: {loss.item():.4f}')

# Teste final
print("\nResultado depois do treino:")
print(torch.round(model(inputs)).detach())
print("Saídas reais:")
print(targets)
