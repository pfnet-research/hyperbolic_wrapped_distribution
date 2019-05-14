from chainer import distribution
from chainer.functions.math import exponential

from lib.distributions.bijector import Bijector, ComposeBijector


class TransformedDistribution(distribution.Distribution):

    """Transformed Distribution.

    `TransformedDistribution` is continuous probablity distribution
    transformed from arbitrary continuous distribution by bijective
    (invertible) function. By using this, we can use flexible distribution
    as like Normalizing Flow.

    Args:
        base_distribution(:class:`~chainer.Distribution`): Arbitrary continuous
        distribution.
        bijector(:class:`~chainer.distributions.Bijector`): Bijective
        (invertible) function.
    """

    def __init__(self, base_distribution, bijector):
        self.base_distribution = base_distribution
        self.bijector = bijector

        self.base_distribution = base_distribution
        if isinstance(bijector, Bijector):
            self.bijector = bijector
        elif isinstance(bijector, list):
            if not all(isinstance(t, Bijector) for t in bijector):
                raise ValueError(
                    "bijector must be a Bijector or a list of Bijectors")
            self.bijector = ComposeBijector(bijector)
        else:
            raise ValueError(
                "bijector must be a Bijector or list, but was {}".format(
                    bijector))
        shape = self.base_distribution.batch_shape \
            + self.base_distribution.event_shape
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

    def log_prob(self, x):
        invx = self.bijector.inverse(x)
        return self.base_distribution.log_prob(invx) \
            - self.bijector.log_det_jacobian(invx, x)

    def sample_n(self, n):
        noise = self.base_distribution.sample_n(n)
        return self.bijector.forward(noise)
