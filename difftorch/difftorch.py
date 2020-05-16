import torch
from . import util


# Value and transposed-Jacobian-vector (vector-Jacobian) product of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f at which f(x) and the transposed-Jacobian-vector product is evaluated
# v: vector in the output domain of f
def fjacobianTv(f, x, v):
    if torch.is_tensor(x) and torch.is_tensor(v):
        if x.dim() == 1 and v.dim() == 1:
            # util.check_vector(x, 'x')
            # util.check_vector(v, 'v')
            x = x.clone().requires_grad_()
            z = f(x)
            # util.check_vector(z, 'f(x)')
            util.check_same_shape(z, 'z', v, 'v')
            z.backward(v)
            return z, x.grad
    return generic_fjacobianTv(f, x, v)


# Transposed-Jacobian-vector (vector-Jacobian) product of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f at which the transposed-Jacobian-vector product is evaluated
# v: vector in the output domain of f
def jacobianTv(f, x, v):
    return fjacobianTv(f, x, v)[1]


# A version of fjacobianTv that supports functions f of multiple Tensor arguments and with multiple Tensor outputs
# f: a function that takes as input a Tensor, list of Tensors or tuple of Tensors, and outputs a Tensor, list of Tensors or tuple of Tensors
# x: a Tensor, list of Tensors or tuple of Tensors in the input domain of f
# v: a Tensor, list of Tensors or tuple of Tensors in the output domain of f
def generic_fjacobianTv(f, x, v):
    xx = util.unflatten_as(util.flatten(x), x)
    z = f(xx) if torch.is_tensor(xx) else f(*xx)

    def genericf(xx):
        xxx = util.unflatten_as(xx, x)
        ff = f(xxx) if torch.is_tensor(xxx) else f(*xxx)
        return util.flatten(ff)

    return z, util.unflatten_as(jacobianTv(genericf, util.flatten(x), util.flatten(v)), x)


# A version of jacobianTv that supports functions f of multiple Tensor arguments and with multiple Tensor outputs
# f: a function that takes as input a Tensor, list of Tensors or tuple of Tensors, and outputs a Tensor, list of Tensors or tuple of Tensors
# x: a Tensor, list of Tensors or tuple of Tensors in the input domain of f
# v: a Tensor, list of Tensors or tuple of Tensors in the output domain of f
def generic_jacobianTv(f, x, v):
    return generic_fjacobianTv(f, x, v)[1]


# Value and Jacobian of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f at which f(x) and the Jacobian is evaluated
def fjacobian(f, x):
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
    return z, torch.stack(j)


# Jacobian of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f at which the Jacobian is evaluated
def jacobian(f, x):
    return fjacobian(f, x)[1]


# Value and Jacobian-vector product of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f at which f(x) and the Jacobian-vector product is evaluated
# v: vector in the input domain of f
def fjacobianv(f, x, v):
    # Uses reverse-mode autodiff because forward-mode is not available in PyTorch
    # Another (and potentially faster) alternative is to use the "double backwards trick"
    if torch.is_tensor(x) and torch.is_tensor(v):
        if x.dim() == 1 and v.dim() == 1:
            # util.check_vector(x, 'x')
            # util.check_vector(v, 'v')
            util.check_same_shape(x, 'x', v, 'v')
            z, j = fjacobian(f, x)
            return z, torch.matmul(j, v)
    return generic_fjacobianv(f, x, v)


# Jacobian-vector product of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f at which the Jacobian-vector product is evaluated
# v: vector in the input domain of f
def jacobianv(f, x, v):
    return fjacobianv(f, x, v)[1]


# A version of fjacobianv that supports functions f of multiple Tensor arguments and with multiple Tensor outputs
# f: a function that takes as input a Tensor, list of Tensors or tuple of Tensors, and outputs a Tensor, list of Tensors or tuple of Tensors
# x: a Tensor, list of Tensors or tuple of Tensors in the input domain of f
# v: a Tensor, list of Tensors or tuple of Tensors in the input domain of f
def generic_fjacobianv(f, x, v):
    xx = util.unflatten_as(util.flatten(x), x)
    z = f(xx) if torch.is_tensor(xx) else f(*xx)

    def genericf(xx):
        xxx = util.unflatten_as(xx, x)
        ff = f(xxx) if torch.is_tensor(xxx) else f(*xxx)
        return util.flatten(ff)

    return z, util.unflatten_as(jacobianv(genericf, util.flatten(x), util.flatten(v)), z)


# A version of jacobianv that supports functions f of multiple Tensor arguments and with multiple Tensor outputs
# f: a function that takes as input a Tensor, list of Tensors or tuple of Tensors, and outputs a Tensor, list of Tensors or tuple of Tensors
# x: a Tensor, list of Tensors or tuple of Tensors in the input domain of f
# v: a Tensor, list of Tensors or tuple of Tensors in the input domain of f
def generic_jacobianv(f, x, v):
    return generic_fjacobianv(f, x, v)[1]


# Value and derivative of a scalar-to-scalar function
# f: scalar-to-scalar function
# x: scalar argument to f at which f(x) and the derivative is evaluated
def fdiff(f, x):
    util.check_scalar(x, 'x')
    x = x.clone().requires_grad_()
    z = f(x)
    util.check_scalar(z, 'f(x)')
    z.backward()
    return z, x.grad


# Derivative of a scalar-to-scalar function
# f: scalar-to-scalar function
# x: scalar argument to f at which the derivative is evaluated
def diff(f, x):
    return fdiff(f, x)[1]


# Value and gradient of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f at which f(x) and the gradient is evaluated
def fgrad(f, x):
    util.check_vector(x, 'x')
    x = x.clone().requires_grad_()
    z = f(x)
    util.check_scalar(z, 'f(x)')
    z.backward()
    return z, x.grad


# Gradient of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f at which the gradient is evaluated
def grad(f, x):
    return fgrad(f, x)[1]


# Value and gradient-vector product (directional derivative) of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f at which f(x) and the gradient-vector product is evaluated
# v: vector in the input domain of f
def fgradv(f, x, v):
    # Uses reverse-mode autodiff because forward-mode is not available in PyTorch
    util.check_vector(x, 'x')
    util.check_vector(v, 'v')
    z, g = fgrad(f, x)
    return z, torch.dot(g, v)


# Gradient-vector product (directional derivative) of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f at which the gradient-vector product is evaluated
# v: vector in the input domain of f
def gradv(f, x, v):
    return fgradv(f, x, v)[1]


# Value and Hessian-vector product of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f at which f(x) and the Hessian-vector product is evaluated
# v: vector in the input domain of f
def fhessianv(f, x, v):
    util.check_vector(x, 'x')
    x = x.clone().requires_grad_()
    z = f(x)
    util.check_scalar(z, 'f(x)')
    g, = torch.autograd.grad(z, x, create_graph=True, allow_unused=True)
    hv, = torch.autograd.grad(g, x, grad_outputs=v, allow_unused=True)
    return z, hv


# Hessian-vector product of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f at which the Hessian-vector product is evaluated
# v: vector in the input domain of f
def hessianv(f, x, v):
    return fhessianv(f, x, v)[1]


# Value and Hessian of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f at which f(x) and the Hessian is evaluated
def fhessian(f, x):
    util.check_vector(x, 'x')
    z0, hv0 = fhessianv(f, x, util.onehot_like(x, 0))
    h = [hv0]
    for i in range(1, x.nelement()):
        v = util.onehot_like(x, i)
        h.append(hessianv(f, x, v))
    return z0, torch.stack(h)


# Hessian of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f at which Hessian is evaluated
def hessian(f, x):
    return fhessian(f, x)[1]


# Value and Laplacian of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f at which f(x) and the Laplacian is evaluated
def flaplacian(f, x):
    z, h = fhessian(f, x)
    return z, h.trace()


# Laplacian of a vector-to-scalar function
# f: vector-to-scalar function
# x: vector argument to f at which the Laplacian is evaluated
def laplacian(f, x):
    return flaplacian(f, x)[1]


# Value and curl of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f at which f(x) and the curl is evaluated
def fcurl(f, x):
    z, j = fjacobian(f, x)
    if j.shape[0] != 3 or j.shape[1] != 3:
        raise RuntimeError('f must have a three-by-three Jacobian')
    return z, torch.stack([j[2, 1] - j[1, 2], j[0, 2] - j[2, 0], j[1, 0] - j[0, 1]])


# Curl of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f at which the curl is evaluated
def curl(f, x):
    return fcurl(f, x)[1]


# Value and divergence of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f at which f(x) and the divergence is evaluated
def fdiv(f, x):
    z, j = fjacobian(f, x)
    if j.shape[0] != j.shape[1]:
        raise RuntimeError('f must have a square Jacobian')
    return z, j.trace()


# Divergence of a vector-to-vector function
# f: vector-to-vector function
# x: vector argument to f at which the divergence is evaluated
def div(f, x):
    return fdiv(f, x)[1]


# Alternative name for gradv
def gvp(f, x, v): return gradv(f, x, v)


# Alternative name for grad
def g(f, x): return grad(f, x)


# Alternative name for hessian
def h(f, x): return hessian(f, x)


# Alternative name for hessianv
def hvp(f, x, v): return hessianv(f, x, v)


# Alternative name for jacobian
def j(f, x): return jacobian(f, x)


# Alternative name for jacobianv
def jvp(f, x, v): return jacobianv(f, x, v)


# Alternative name for jacobianTv
def vjp(f, x, v): return jacobianTv(f, x, v)


# Alternative name for fgradv
def fgvp(f, x, v): return fgradv(f, x, v)


# Alternative name for fgrad
def fg(f, x): return fgrad(f, x)


# Alternative name for fhessian
def fh(f, x): return fhessian(f, x)


# Alternative name for fhessianv
def fhvp(f, x, v): return fhessianv(f, x, v)


# Alternative name for fhessian
def fj(f, x): return fjacobian(f, x)


# Alternative name for fjacobianv
def fjvp(f, x, v): return fjacobianv(f, x, v)


# Alternative name for fjacobianTv
def fvjp(f, x, v): return fjacobianTv(f, x, v)
