import chainer
from chainer.backends import cuda
import chainer.functions as F
# from chainer.utils import cache  # cache is only works for version >= 6

from lib.distributions import bijector, transformed_distribution
from lib import functions, xp_functions
from lib.miscs.utils import eps


class ConcatFirstAxisBijector(bijector.Bijector):

    def __init__(self, value, **kwargs):
        super(ConcatFirstAxisBijector, self).__init__(**kwargs)
        self.value = value

    def _forward(self, x):
        num_expand = len(x.shape) - len(self.value.shape)
        value = F.broadcast_to(
            F.reshape(self.value, (1,) * num_expand + self.value.shape),
            x.shape[:-1] + (1,))
        return F.concat((value, x), axis=-1)

    def _inverse(self, y):
        return y[..., 1:]

    def _log_det_jacobian(self, x, y):
        xp = cuda.get_array_module(x, y)
        return xp.zeros_like(x.array)


class ParallelTransportBijector(bijector.Bijector):

    event_dim = 1

    def __init__(self, from_, to_, **kwargs):
        super(ParallelTransportBijector, self).__init__(**kwargs)
        self.from_ = from_
        self.to_ = to_

    def _forward(self, x):
        num_expand = len(x.shape) - len(self.from_.shape)
        from_ = F.broadcast_to(F.reshape(
            self.from_, (1,) * num_expand + self.from_.shape), x.shape)
        to_ = F.broadcast_to(F.reshape(
            self.to_, (1,) * num_expand + self.to_.shape), x.shape)
        return functions.parallel_transport(x, from_, to_)

    def _inverse(self, y):
        num_expand = len(y.shape) - len(self.from_.shape)
        from_ = F.broadcast_to(F.reshape(
            self.from_, (1,) * num_expand + self.from_.shape), y.shape)
        to_ = F.broadcast_to(F.reshape(
            self.to_, (1,) * num_expand + self.to_.shape), y.shape)
        return functions.inv_parallel_transport(y, to_, from_)

    def _log_det_jacobian(self, x, y):
        xp = cuda.get_array_module(x, y)
        logdet = xp.zeros((1,), dtype=x.dtype)

        shape = x.shape[:-1]
        num_expand = len(shape) - len(logdet.shape)
        logdet = F.broadcast_to(
            F.reshape(logdet, (1,) * num_expand + logdet.shape), shape)
        return logdet


class ExpMapBijector(bijector.Bijector):

    event_dim = 1

    def __init__(self, loc, **kwargs):
        super(ExpMapBijector, self).__init__(**kwargs)
        self.loc = loc

    def _forward(self, x):
        return functions.exponential_map(self.loc, x)

    def _inverse(self, y):
        return functions.inv_exponential_map(self.loc, y)

    def _log_det_jacobian(self, x, y):
        r = F.sqrt(functions.clamp(functions.lorentzian_product(x, x), eps))
        d = x / r[..., None]
        dim = d.shape[-1]
        logdet = (dim - 2) * F.log(F.sinh(r) / r)

        return logdet


class HyperbolicWrapped(
        transformed_distribution.TransformedDistribution):

    def __init__(self, base_distribution, loc, **kwargs):
        self.__loc = loc
        origin = xp_functions.make_origin_like(loc.array)

        bijector = [
            ConcatFirstAxisBijector(origin[..., 0:1] * 0, cache=True),
            ParallelTransportBijector(origin[...], loc, cache=True),
            ExpMapBijector(loc, cache=True)
        ]

        super(HyperbolicWrapped, self).__init__(
            base_distribution, bijector)

        shape = self.base_distribution.batch_shape \
            + self.base_distribution.event_shape
        shape = shape[:-1] + (shape[-1] + 1,)
        event_dim = max(
            [len(self.base_distribution.event_shape), self.bijector.event_dim])
        self.__batch_shape = shape[:len(shape) - event_dim]
        self.__event_shape = shape[len(shape) - event_dim:]

    @property
    def batch_shape(self):
        return self.__batch_shape

    @property
    def event_shape(self):
        return self.__event_shape

    # @cache.cached_property
    @property
    def loc(self):
        return chainer.as_variable(self.__loc)

    @property
    def _is_gpu(self):
        return isinstance(self.loc.array, cuda.ndarray)

    # @cache.cached_property
    @property
    def mean(self):
        return self.loc

    @property
    def params(self):
        return {'loc': self.loc, **self.base_distribution.params}

    @property
    def support(self):
        return 'lorentz'
