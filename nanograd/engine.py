import math

class Value:
  """ stores a single scalar value and its gradient """
  def __init__(self,data,_children=(),_op='',label=""): 
    self.data=data
    self.grad=0
    self._backward=lambda:None
    self._prev=set(_children)
    self._op=_op
    self.label=label

  def __repr__(self):
    return f"Value(data={self.data})"

  def __rmul__(self,other): #other* self
    return self * other

  def __radd__(self,other):
    return self + other

  def __mul__(self,other):
    other = other if isinstance(other, Value) else Value(other)
    out=Value(self.data * other.data,(self,other),'*')

    def _backward():
      self.grad+= out.grad * other.data
      other.grad+= out.grad * self.data

    out._backward= _backward
    return out

  def __pow__(self,other):
    assert isinstance(other,(int,float)), " only supporting int and float"
    out = Value(self.data**other,(self,),f'**{other}')

    def _backward():
      self.grad += other*((self.data)**(other-1)) * out.grad

    out._backward=_backward
    return out

  def __truediv__(self,other): #self/other
    return self * other**-1

  def tanh(self):
    x= self.data
    t= (math.exp(2*x)-1)/(math.exp(2*x)+1)
    out = Value(t,(self,),'tanh')
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward= _backward

    return out

  def __neg__(self): # -self
    return self * -1

  def __sub__(self,other): #self-other
    return self + (-other)

  def __add__(self,other):
    other = other if isinstance(other,Value) else Value(other)
    out=Value(self.data+other.data,(self,other),'+')

    def _backward():
      self.grad+= 1.0 * out.grad
      other.grad+= 1.0 * out.grad

    out._backward= _backward
    return out

  def exp(self):
    x= self.data
    out = Value(math.exp(x),(self,),'exp')

    def _backward():
      self.grad+= out.data * out.grad
    out._backward= _backward
    return out

  def reLU(self):
    out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out
  
  def __rtruediv__(self, other):  # other/self
    return other * self**-1
  
  def backward(self):
    topo=[]
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad=1.0
    for node in reversed(topo):
      node._backward()