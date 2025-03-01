import random
from nanograd.engine import Value
import math

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
class Neuron:
  def __init__(self,nin):  #nin number of inputs in the neuron
    self.w= [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b= Value(random.uniform(-1,1))

  def __call__(self, x):
      # Compute the dot product of weights and inputs, then add bias
      act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
      out = act.tanh()
      return out
  def parameters(self):
      return self.w + [self.b]

class Layer:
  def __init__(self,nin,nout):
    self.neurons=[Neuron(nin) for _ in range(nout)]

  def __call__(self,x):
    outs= [n(x) for n in self.neurons ]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
    # params= []
    # for neuron in self.neurons:
    #   ps = neuron.parameters()
    #   params.extend(ps)
    # return params

class MLP:
  def __init__(self,nin,nouts):
    sz=[nin]+nouts
    self.layers=[Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

  def __call__(self,x):
    for layer in self.layers:
      x=layer(x)
    return x
  
  def parameters(self):
    return [p for neuron in self.layers for p in neuron.parameters()]
