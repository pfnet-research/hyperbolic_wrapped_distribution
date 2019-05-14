import numbers

from chainer.backend import cuda
from chainer.functions.array import broadcast
from chainer.functions.array import reshape
from chainer.functions.math import basic_math
from chainer.functions.math import exponential
from chainer.functions.math import sum as sum_mod


def _sum_rightmost(value, dim):
    """Sum out `dim` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return sum_mod.sum(reshape.reshape(value, required_shape), axis=-1)


class Bijector(object):

    """Interface of Bijector.

    `Bijector` is implementation of bijective (invertible) function that is
    used by `TransformedDistribution`. The three method `_forward`, `_inverse`
    and `_log_det_jacobian` have to be defined in inhereted class.
    """

    event_dim = 0

    def __init__(self, cache=False):
        self.cache = cache
        if self.cache:
            self.cache_x_y = (None, None)

    def forward(self, x):
        if self.cache:
            old_x, old_y = self.cache_x_y
            if x is old_x:
                return old_y
            y = self._forward(x)
            self.cache_x_y = x, y
        else:
            y = self._forward(x)
        return y

    def inverse(self, y):
        if self.cache:
            old_x, old_y = self.cache_x_y
            if y is old_y:
                return old_x
            x = self._inverse(y)
            self.cache_x_y = x, y
        else:
            x = self._inverse(y)
        return x

    def log_det_jacobian(self, x, y):
        return self._log_det_jacobian(x, y)

    def _forward(self, x):
        """Forward computation

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Data points in the domain of the
            based distribution.

        Returns:
            ~chainer.Variable: Transformed data points in the domain of the
            transformed distribution.
        """
        raise NotImplementedError

    def _inverse(self, y):
        """Inverse computation

        Args:
            y(:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
            :class:`cupy.ndarray`): Data points in the domain of the
            transformed distribution.

        Returns:
            ~chainer.Variable: Transformed data points in the domain of the
            based distribution.
        """
        raise NotImplementedError

    def _log_det_jacobian(self, x, y):
        """Computes the log det jacobian :math:`log |dy/dx|` given input and
        output.

        Args:
            x(:class:`~chainer.Variable` or :class:`numpy.ndarray` or
                :class:`cupy.ndarray`): Data points in the domain of the
                based distribution.
            y(:class:`~chainer.Variable` or :class:`numpy.ndarray` or
                :class:`cupy.ndarray`): Data points in the codomain of the
                based distribution.

        Returns:
            ~chainer.Variable: Log-Determinant of Jacobian matrix in given
                input and output.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ComposeBijector(Bijector):
    """
    Composes multiple bijectors in a chain.
    The bijectors being composed are responsible for caching.

    Args:
        parts (list of :class:`Bijector`): A list of transforms to compose.
    """

    def __init__(self, parts):
        super(ComposeBijector, self).__init__()
        self.parts = parts

    def __getitem__(self, index):
        return self.parts[index]

    @property
    def event_dim(self):
        return max(p.event_dim for p in self.parts) if self.parts else 0

    def _forward(self, x):
        for part in self.parts:
            x = part.forward(x)
        return x

    def _inverse(self, y):
        x = y
        for part in reversed(self.parts):
            x = part.inverse(x)
        return x

    def _log_det_jacobian(self, x, y):
        if not self.parts:
            xp = cuda.get_array_module(x, y)
            return xp.zeros_like(x.array)
        result = 0
        for part in self.parts:
            y = part.forward(x)
            result = result + _sum_rightmost(
                part.log_det_jacobian(x, y),
                self.event_dim - part.event_dim)
            x = y
        return result

    def __repr__(self):
        fmt_string = self.__class__.__name__ + '(\n    '
        fmt_string += ',\n    '.join([p.__repr__() for p in self.parts])
        fmt_string += '\n)'
        return fmt_string


identity_transform = ComposeBijector([])


class ExpBijector(Bijector):
    """ExpBijector.

    Transform via the mapping :math:`y = \\exp(x)`.
    """

    def _forward(self, x):
        return exponential.exp(x)

    def _inverse(self, y):
        return exponential.log(y)

    def _log_det_jacobian(self, x, y):
        return x


class AffineBijector(Bijector):
    """Affine Bijector.

    Transform via the pointwise affine mapping :math:`y = \\text{loc} +
    \\text{scale} \\times x`.

    Args:
        loc (Tensor or float): Location parameter.
        scale (Tensor or float): Scale parameter.
        event_dim (int): Optional size of `event_shape`. This should be zero
            for univariate random variables, 1 for distributions over vectors,
            2 for distributions over matrices, etc.
    """

    def __init__(self, loc, scale, event_dim=0, cache_size=0):
        super(AffineBijector, self).__init__(cache_size=cache_size)
        self.loc = loc
        self.scale = scale
        self.event_dim = event_dim

    def _forward(self, x):
        return self.loc + self.scale * x

    def _inverse(self, y):
        return (y - self.loc) / self.scale

    def _log_det_jacobian(self, x, y):
        shape = x.shape
        scale = self.scale
        if isinstance(scale, numbers.Number):
            xp = cuda.get_array_module(x, y)
            result = exponential.log(basic_math.absolute(scale)) \
                * xp.ones(shape, dtype=x.dtype)
        else:
            result = exponential.log(basic_math.absolute(scale))
        if self.event_dim:
            result_size = result.shape[:-self.event_dim] + (-1,)
            result = sum_mod.sum(result.view(result_size), axis=-1)
            shape = shape[:-self.event_dim]
        return broadcast.broadcast_to(result, shape)
