import torch


def check_vector(x, name):
    if not torch.is_tensor(x):
        raise RuntimeError('{} needs to be a Tensor'.format(name))
    if x.dim() != 1:
        raise RuntimeError('{} needs to be a vector (one-dimensional Tensor)'.format(name))


def check_scalar(x, name):
    if not torch.is_tensor(x):
        raise RuntimeError('{} needs to be a Tensor'.format(name))
    if x.dim() != 0:
        raise RuntimeError('{} needs to be a scalar (zero-dimensional Tensor)'.format(name))


def onehot_like(x, i):
    ret = torch.zeros_like(x)
    ret[i] = 1.
    return ret


# Transposed-Jacobian-vector (vector-Jacobian) product of vector-to-vector function f, evaluated at x, with vector v
def jacobianTv(f, x, v):
    check_vector(x, 'x')
    check_vector(v, 'v')
    x = x.clone().requires_grad_()
    z = f(x)
    check_vector(z, 'z')
    z.backward(v)
    return x.grad


# Jacobian of vector-to-vector function f, evaluated at x
def jacobian(f, x):
    check_vector(x, 'x')
    x = x.clone().requires_grad_()
    z = f(x)
    check_vector(z, 'z')
    j = []
    for i in range(z.nelement()):
        x.grad = None
        v = onehot_like(z, i)
        z.backward(v, retain_graph=True)
        j.append(x.grad)
    return torch.stack(j)


# Jacobian-vector product of vector-to-vector function f, evaluated at x, with vector v
def jacobianv(f, x, v):
    # Uses reverse-mode autodiff because forward-mode is not available in PyTorch
    # Another (and potentially faster) alternative is to use the "double backwards trick"
    check_vector(x, 'x')
    check_vector(v, 'v')
    j = jacobian(f, x)
    return torch.matmul(j, v)


# Derivative of scalar-to-scalar function f, evaluated at x
def diff(f, x):
    check_scalar(x, 'x')
    x = x.clone().requires_grad_()
    z = f(x)
    check_scalar(z, 'z')
    z.backward()
    return x.grad


# Gradient of vector-to-scalar function f, evaluated at x
def grad(f, x):
    check_vector(x, 'x')
    x = x.clone().requires_grad_()
    z = f(x)
    check_scalar(z, 'z')
    z.backward()
    return x.grad


# Gradient-vector product (directional derivative) of vector-to-scalar function f, evaluated at x, with vector v
def gradv(f, x, v):
    check_vector(x, 'x')
    check_vector(v, 'v')
    g = grad(f, x)
    return torch.dot(g, v)


# Hessian-vector product of vector-to-scalar function f, evaluated at x, with vector v
def hessianv(f, x, v):
    x = x.clone().requires_grad_()
    z = f(x)
    check_scalar(z, 'z')
    g, = torch.autograd.grad(z, x, create_graph=True, allow_unused=True)
    hv, = torch.autograd.grad(g, x, grad_outputs=v, allow_unused=True)
    return hv


# Hessian of vector-to-scalar function f, evaluated at x
def hessian(f, x):
    check_vector(x, 'x')
    h = []
    for i in range(x.nelement()):
        v = onehot_like(x, i)
        h.append(hessianv(f, x, v))
    return torch.stack(h)


# Laplacian of vector-to-scalar function f, evaluated at x
def laplacian(f, x):
    return hessian(f, x).trace()


# Curl of vector-to-vector function f, evaluated at x
def curl(f, x):
    j = jacobian(f, x)
    if j.shape[0] != 3 or j.shape[1] != 3:
        raise RuntimeError('f should have a three-by-three Jacobian')
    return torch.stack([j[2, 1] - j[1, 2], j[0, 2] - j[2, 0], j[1, 0] - j[0, 1]])


# Divergence of vector-to-vector function f, evaluated at x
def div(f, x):
    j = jacobian(f, x)
    if j.shape[0] != j.shape[1]:
        raise RuntimeError('f must have a square Jacobian')
    return j.trace()
