import math

from chainer.backends import cuda

from lib.miscs.utils import eps


def _softplus_inverse(x):
    return math.log(math.expm1(x))


def lorentzian_product(u, v=None, keepdims=False):
    if v is None:
        v = u
    xp = cuda.get_array_module(u, v)
    uv = u * v
    uv[..., 0] *= -1
    return xp.sum(uv, axis=-1, keepdims=keepdims)


def pseudo_polar_projection(x):
    xp = cuda.get_array_module(x)
    r = xp.sqrt(xp.sum(xp.square(x), axis=-1, keepdims=True))
    d = x / xp.broadcast_to(xp.maximum(r, eps), x.shape)

    r_proj = xp.cosh(r)
    d_proj = xp.broadcast_to(xp.sinh(r), d.shape) * d
    x_proj = xp.concatenate((r_proj, d_proj), axis=-1)

    return x_proj


def exponential_map(x, v):
    xp = cuda.get_array_module(x, v)
    vnorm = xp.sqrt(xp.maximum(lorentzian_product(v, keepdims=True), eps))
    return xp.cosh(vnorm) * x + xp.sinh(vnorm) * v / vnorm


def inv_exponential_map(x, z):
    xp = cuda.get_array_module(x, z)
    alpha = -lorentzian_product(x, z, keepdims=True)
    C = xp.arccosh(-lorentzian_product(x, z, keepdims=True)) \
        / xp.sqrt(xp.maximum(alpha ** 2 - 1, eps))
    return C * (z - alpha * x)


def hyperbolic_interpolate(x, y, ratio=0.5):
    v = inv_exponential_map(x, y)
    return exponential_map(x, (1 - ratio) * v)


def hyperbolic_average(data, keepdims=False, shuffle=False):
    xp = cuda.get_array_module(data)
    if shuffle:
        indices = xp.arange(len(data))
        xp.random.shuffle(indices)
        data = data[indices]
    centroid = data[0]
    for n in range(1, len(data)):
        centroid = hyperbolic_interpolate(centroid, data[n], ratio=n / (n + 1))
    if keepdims:
        centroid = centroid[None, ...]
    return centroid


def make_origin_like(x):
    xp = cuda.get_array_module(x)
    origin = xp.zeros_like(x)
    origin[..., 0] = 1
    return origin
