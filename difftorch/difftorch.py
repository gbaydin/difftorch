import torch
from . import util


# Transposed-Jacobian-vector (vector-Jacobian) product of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f(x) at which the transposed-Jacobian-vector product is evaluated
# v: vector in the output domain of f
def jacobianTv(f, x, v):
    util.check_vector(x, 'x')
    util.check_vector(v, 'v')
    x = x.clone().requires_grad_()
    z = f(x)
    util.check_vector(z, 'f(x)')
    z.backward(v)
    return x.grad


# Jacobian of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f(x) at which the Jacobian is evaluated
def jacobian(f, x):
    util.check_vector(x, 'x')
    x = x.clone().requires_grad_()
    z = f(x)
    util.check_vector(z, 'f(x)')
    j = []
    for i in range(z.nelement()):
        x.grad = None
        v = util.onehot_like(z, i)
        z.backward(v, retain_graph=True)
        j.append(x.grad)
    return torch.stack(j)


# Jacobian-vector product of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f(x) at which the Jacobian-vector product is evaluated
# v: vector in the input domain of f
def jacobianv(f, x, v):
    # Uses reverse-mode autodiff because forward-mode is not available in PyTorch
    # Another (and potentially faster) alternative is to use the "double backwards trick"
    util.check_vector(x, 'x')
    util.check_vector(v, 'v')
    j = jacobian(f, x)
    return torch.matmul(j, v)


# Derivative of a scalar-to-scalar function
# f: scalar-to-scalar function
# x: scalar argument to f(x) at which the derivative is evaluated
def diff(f, x):
    util.check_scalar(x, 'x')
    x = x.clone().requires_grad_()
    z = f(x)
    util.check_scalar(z, 'f(x)')
    z.backward()
    return x.grad


# Gradient of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f(x) at which the gradient is evaluated
def grad(f, x):
    util.check_vector(x, 'x')
    x = x.clone().requires_grad_()
    z = f(x)
    util.check_scalar(z, 'f(x)')
    z.backward()
    return x.grad


# Gradient-vector product (directional derivative) of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f(x) at which the gradient-vector product is evaluated
# v: vector in the input domain of f
def gradv(f, x, v):
    # Uses reverse-mode autodiff because forward-mode is not available in PyTorch
    util.check_vector(x, 'x')
    util.check_vector(v, 'v')
    g = grad(f, x)
    return torch.dot(g, v)


# Hessian-vector product of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f(x) at which the Hessian-vector product is evaluated
# v: vector in the input domain of f
def hessianv(f, x, v):
    util.check_vector(x, 'x')
    x = x.clone().requires_grad_()
    z = f(x)
    util.check_scalar(z, 'f(x)')
    g, = torch.autograd.grad(z, x, create_graph=True, allow_unused=True)
    hv, = torch.autograd.grad(g, x, grad_outputs=v, allow_unused=True)
    return hv


# Hessian of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f(x) at which the Hessian is evaluated
def hessian(f, x):
    util.check_vector(x, 'x')
    h = []
    for i in range(x.nelement()):
        v = util.onehot_like(x, i)
        h.append(hessianv(f, x, v))
    return torch.stack(h)


# Laplacian of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f(x) at which the Laplacian is evaluated
def laplacian(f, x):
    return hessian(f, x).trace()


# Curl of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f(x) at which the curl is evaluated
def curl(f, x):
    j = jacobian(f, x)
    if j.shape[0] != 3 or j.shape[1] != 3:
        raise RuntimeError('f must have a three-by-three Jacobian')
    return torch.stack([j[2, 1] - j[1, 2], j[0, 2] - j[2, 0], j[1, 0] - j[0, 1]])


# Divergence of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f(x) at which the divergence is evaluated
def div(f, x):
    j = jacobian(f, x)
    if j.shape[0] != j.shape[1]:
        raise RuntimeError('f must have a square Jacobian')
    return j.trace()


# A version of jacobianv that supports functions f of multiple Tensor arguments and multiple Tensor outputs
# f: a function that takes as input a Tensor, list of Tensors or tuple of Tensors, and outputs a Tensor, list of Tensors or tuple of Tensors
# x: a Tensor, list of Tensors or tuple of Tensors in the input domain of f
# v: a Tensor, list of Tensors or tuple of Tensors in the input domain of f
def generic_jacobianv(f, x, v):
    return util.unflatten_as(jacobianv(lambda xx: util.flatten(f(*util.unflatten_as(xx, x))), util.flatten(x), util.flatten(v)), f(*x))


# A version of jacobianTv that supports functions f of multiple Tensor arguments and multiple Tensor outputs
# f: a function that takes as input a Tensor, list of Tensors or tuple of Tensors, and outputs a Tensor, list of Tensors or tuple of Tensors
# x: a Tensor, list of Tensors or tuple of Tensors in the input domain of f
# v: a Tensor, list of Tensors or tuple of Tensors in the output domain of f
def generic_jacobianTv(f, x, v):
    return util.unflatten_as(jacobianTv(lambda xx: util.flatten(f(*util.unflatten_as(xx, x))), util.flatten(x), util.flatten(v)), x)
