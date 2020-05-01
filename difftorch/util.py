import torch
import typing


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


def unzip(lst):
    return zip(*lst)


def check_tensor(value):
    if not torch.is_tensor(value):
        raise ValueError('Expecting value to be a tensor')


def check_list_or_tuple_of_tensors(value):
    if not (isinstance(value, typing.List) or isinstance(value, typing.Tuple)):
        raise ValueError('Expecting value to be a list or tuple')
    for v in value:
        check_tensor(v)


def flatten(tensors):
    if torch.is_tensor(tensors):
        return tensors.view(-1)
    check_list_or_tuple_of_tensors(tensors)
    return torch.cat([t.view(-1) for t in tensors])


def unflatten_as(tensor, tensors):
    if torch.is_tensor(tensors):
        return tensor.view(tensors.shape)
    check_tensor(tensor)
    check_list_or_tuple_of_tensors(tensors)
    shapes = [t.shape for t in tensors]
    nelements = [t.nelement() for t in tensors]
    ts = torch.split(tensor.view(-1), nelements)
    return [ts[i].view(shapes[i]) for i in range(len(tensors))]
