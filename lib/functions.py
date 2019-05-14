from chainer.backends import cuda
import chainer.functions as F

from lib.miscs.utils import eps


def clamp(a, a_min):
    return F.relu(a - a_min) + a_min


def negate_first_index(x):
    x0, xrest = F.split_axis(x, indices_or_sections=(1,), axis=-1)
    return F.concat((x0 * -1, xrest), axis=-1)


def lorentzian_product(u, v=None, keepdims=False):
    if v is None:
        v = u
    uv = u * v
    return F.sum(negate_first_index(uv), axis=-1, keepdims=keepdims)


def lorentz_distance(u, v, keepdims=False):
    negprod = -lorentzian_product(u, v, keepdims=keepdims)
    z = F.sqrt(negprod**2 - 1)
    return F.log(negprod + z)


def exponential_map(x, v):
    vnorm = F.sqrt(clamp(lorentzian_product(v, keepdims=True), eps))
    return F.cosh(vnorm) * x + F.sinh(vnorm) * v / vnorm


def inv_exponential_map(x, z):
    alpha = -lorentzian_product(x, z, keepdims=True)
    C = lorentz_distance(x, z, keepdims=True) \
        / F.sqrt(clamp(alpha ** 2 - 1, eps))
    return C * (z - alpha * x)


def pseudo_polar_projection(x):
    r = F.sqrt(F.sum(F.square(x), axis=-1, keepdims=True))
    d = x / F.broadcast_to(clamp(r, eps), x.shape)

    r_proj = F.cosh(r)
    d_proj = F.broadcast_to(F.sinh(r), d.shape) * d
    x_proj = F.concat((r_proj, d_proj), axis=-1)

    return x_proj


def inv_pseudo_polar_projection(z):
    xp = cuda.get_array_module(z)
    origin = xp.zeros(shape=z.shape, dtype=z.dtype)
    origin[..., 0] = 1
    return inv_exponential_map(origin, z)


def parallel_transport(xi, x, y):
    alpha = -lorentzian_product(x, y, keepdims=True)
    coef = lorentzian_product(y, xi, keepdims=True) / (alpha + 1)
    # coef = lorentzian_product(y - alpha * x, xi, keepdims=True) / (alpha + 1)
    return xi + coef * (x + y)


def inv_parallel_transport(xi, x, y):
    return parallel_transport(xi, x, y)


def h2p(x):
    x0, xrest = F.split_axis(x, indices_or_sections=(1,), axis=-1)
    ret = (xrest / F.broadcast_to(1 + x0, xrest.shape))
    return ret


def p2h(x):
    xsqnorm = F.sum(F.square(x), axis=-1, keepdims=True)
    ret = F.concat((1 + xsqnorm, 2 * x), axis=-1)
    return ret / clamp(1 - xsqnorm, eps)
