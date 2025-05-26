import numpy as np

#definir as funcoes iniciais 
def sigmoids(x):
    return 1/ (1+ np.exp(-x))

def sigmoid_deriv(x):
    return x*(1-x)

#dados tester de entrada  (XOR)
inputs = np.array ([
    [0,0],
    [0,1],
    [1,0],
    [1,1] 
])

#dados de saida esperados
expected_output = np.array([
    [0],
    [1],
    [1],
    [0]
])
#inicializar pesos aleatorios
np.random.seed(42)
weight_hidden_imput = np.random.uniform(size=(2,2))
bias_hidden = np.random.uniform(size=(1,2))

weight_output_hidden = np.random.uniform(size=(2,1))
bias_output = np.random.uniform(size=(1,1))

#taxa de aprendizado
learning_rate = 0.1 #ou seja 0.1 é 10% do q ele vai progr bedir por teste iteração

epochs = 10000 #número de iterações

#treinamento da rede neural
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(inputs, weight_hidden_imput) + bias_hidden
    hidden_output = sigmoids(hidden_input)
    final_input = np.dot(hidden_output, weight_output_hidden) + bias_output
    predicted_output = sigmoids(final_input)
    error = expected_output - predicted_output
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")
    # Backpropagation
    d_predicted_output = error * sigmoid_deriv(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weight_output_hidden.T)
    d_hidden_output = error_hidden_layer * sigmoid_deriv(hidden_output)
    # Update weights and biases
    weight_output_hidden += hidden_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weight_hidden_imput += inputs.T.dot(d_hidden_output) * learning_rate
    bias_hidden += np.sum(d_hidden_output, axis=0, keepdims=True) * learning_rate

# testes
print("Saídas previstas após o treinamento:")
print(predicted_output)