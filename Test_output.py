from nanograd import nn
from nanograd.engine import Value
# Create a basic neuron with 3 inputs
neuron = nn.Neuron(3)

# Create a layer of 5 neurons, each taking 10 inputs
layer = nn.Layer(10, 5)

# Create a multi-layer perceptron with 3 inputs, 
# a hidden layer of 4 neurons, and 1 output neuron
mlp = nn.MLP(3, [4, 1])

# Forward pass
x = [Value(1.0), Value(0.5), Value(-1.0)]  # input
output = mlp(x)

# Print the output value
print("Output:", output.data)

# Backward pass for gradients
output.backward()

# Print gradients for each parameter in the MLP
for param in mlp.parameters():
    print(f"Param value: {param.data}, Gradient: {param.grad}")

# Zero gradients before next pass
mlp.zero_grad()