import numpy as np

# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Dados de entrada (XOR)
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Saída esperada
expected_output = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Inicialização dos pesos
np.random.seed(42)
weight_input_hidden = np.random.uniform(size=(2, 2))
bias_hidden = np.random.uniform(size=(1, 2))

weight_hidden_output = np.random.uniform(size=(2, 1))
bias_output = np.random.uniform(size=(1, 1))

# Hiperparâmetros
learning_rate = 0.1
epochs = 10000

# Treinamento
for epoch in range(epochs):
    # Forward
    hidden_input = np.dot(inputs, weight_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weight_hidden_output) + bias_output
    predicted_output = sigmoid(final_input)

    # Erro
    error = expected_output - predicted_output

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Erro médio: {np.mean(np.abs(error)):.4f}")

    # Backpropagation
    d_predicted_output = error * sigmoid_deriv(predicted_output)
    error_hidden = d_predicted_output.dot(weight_hidden_output.T)
    d_hidden_output = error_hidden * sigmoid_deriv(hidden_output)

    # Atualização dos pesos
    weight_hidden_output += hidden_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate

    weight_input_hidden += inputs.T.dot(d_hidden_output) * learning_rate
    bias_hidden += np.sum(d_hidden_output, axis=0, keepdims=True) * learning_rate

# Resultado final
print("\nSaídas previstas após o treinamento:")
print(np.round(predicted_output, 3))
