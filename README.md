## difftorch
[![Build Status](https://travis-ci.org/gbaydin/difftorch.svg?branch=master)](https://travis-ci.org/gbaydin/difftorch)
[![PyPI version](https://badge.fury.io/py/difftorch.svg)](https://badge.fury.io/py/difftorch)

difftorch is a little library providing a functional differentiation API for [PyTorch](https://pytorch.org/). The operations it implements mirror those available in [DiffSharp](https://github.com/DiffSharp/DiffSharp).


## Installation

### Install the latest package
To use the latest version available in [Python Package
Index](https://pypi.org/project/difftorch/), run:

```
pip install difftorch
```

### Install from source
To install from the source, clone this repository and install the difftorch package using:

```
git clone https://github.com/gbaydin/difftorch.git
cd difftorch
pip install .
```
## Examples
```python
import torch
import difftorch

def f_rosenbrock(x):
  x, y = x[0], x[1]
  return (1. - x)**2 + 100. * (y - x**2)**2

def f_spherical_cartesian(x):
    r, theta, phi = x[0], x[1], x[2]
    x, y, z = r*phi.sin()*theta.cos(), r*phi.sin()*theta.sin(), r*phi.cos()
    return torch.stack([x, y, z])
    
# grad
# Gradient of vector-to-scalar function f, evaluated at x
x = torch.randn(2)
g = difftorch.grad(f_rosenbrock, x)
print(x, g)

# gradv
# Gradient-vector product (directional derivative) of vector-to-scalar function f, evaluated at x, with vector v
x, v = torch.randn(2), torch.randn(2)
gv = difftorch.gradv(f_rosenbrock, x, v)
print(x, v, gv)

# jacobian
# Jacobian of vector-to-vector function f, evaluated at x
x = torch.randn(3)
j = difftorch.jacobian(f_spherical_cartesian, x)
print(x, j)

# jacobianv
# Jacobian-vector product of vector-to-vector function f, evaluated at x, with vector v
x, v = torch.randn(3), torch.randn(3)
jv = difftorch.jacobianv(f_spherical_cartesian, x, v)
print(x, v, jv)

# jacobianTv
# Transposed-Jacobian-vector (vector-Jacobian) product of vector-to-vector function f, evaluated at x, with vector v
x, v = torch.randn(3), torch.randn(3)
jv = difftorch.jacobianTv(f_spherical_cartesian, x, v)
print(x, v, jv)

# hessianv
# Hessian-vector product of vector-to-scalar function f, evaluated at x, with vector v
x, v = torch.randn(2), torch.randn(2)
hv = difftorch.hessianv(f_rosenbrock, x, v)
print(x, v, hv)

# hessian
# Hessian of vector-to-scalar function f, evaluated at x
x = torch.randn(2)
h = difftorch.hessian(f_rosenbrock, x)
print(x, h)

# laplacian
# Laplacian of vector-to-scalar function f, evaluated at x
x = torch.randn(2)
l = difftorch.laplacian(f_rosenbrock, x)
print(x, l)

# curl
# Curl of vector-to-vector function f, evaluated at x
x = torch.randn(3)
c = difftorch.curl(f_spherical_cartesian, x)
print(x, c)

# div
# Divergence of vector-to-vector function f, evaluated at x
x = torch.randn(3)
d = difftorch.div(f_spherical_cartesian, x)
print(x, d)
```
