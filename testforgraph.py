from nanograd import nn
from nanograd.engine import Value
from nanograd.utils import draw_dot

# Create computation
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)

y.backward()  

# Now visualize with gradients
dot = draw_dot(y)
dot.render('computation_graph', format='png', cleanup=True)