import numpy

from chainer.training import extension


class Burnin(extension.Extension):

    def __init__(self, attr, burnin_step, c, optimizer=None):
        self._attr = attr
        self._burnin_step = burnin_step
        self._c = c
        self._init = None
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None
        self._fired = False

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        # ensure that _init is set
        if self._init is None:
            self._init = getattr(optimizer, self._attr)

        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def __call__(self, trainer):
        self._t += 1

        optimizer = self._get_optimizer(trainer)
        value = getattr(optimizer, self._attr)
        if self._t <= self._burnin_step:
            value = self._init / self._c
        elif not self._fired:
            value = self._init
            self._fired = True
        self._update_value(optimizer, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, numpy.ndarray):
            self._last_value = self._last_value.item()

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value
