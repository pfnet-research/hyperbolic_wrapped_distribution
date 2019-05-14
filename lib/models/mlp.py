import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I


class MLPRepeat(chainer.ChainList):

    def __init__(
            self, first_layer, n_unit, n_layer, nonlinearity,
            initialW=I.GlorotUniform()):

        super(MLPRepeat, self).__init__()
        with self.init_scope():
            self.nonlinearity = nonlinearity
            self.add_link(first_layer)
            for layer in range(n_layer - 1):
                self.add_link(L.Linear(n_unit, n_unit, initialW=initialW))

    def __call__(self, x, **kwargs):
        for link in self.children():
            pre_activate = link(x, **kwargs)
            x = self.nonlinearity(pre_activate)
        return x


class MLPHead(chainer.Chain):

    def __init__(
            self, n_in, n_hidden, n_out,
            n_layer=1, nonlinearity=F.tanh, initialW=I.GlorotUniform(),
            **kwargs):

        super(EncoderHead, self).__init__()
        self.n_layer = n_layer
        with self.init_scope():
            self.nonlinearity = nonlinearity
            first_layer = L.Linear(n_in, n_hidden)
            self.linears = MLPRepeat(
                first_layer, n_hidden, n_layer, nonlinearity,
                initialW=initialW)
            self.output = L.Linear(n_hidden, n_out, initialW=initialW)

    def forward(self, x, n_batch_axes=1):
        h = self.linears(x, n_batch_axes=n_batch_axes)
        return self.output(h, n_batch_axes=n_batch_axes)


EncoderHead = MLPHead
DecoderHead = MLPHead


def make_encoder_head(data_shape, n_out, **kwargs):
    return EncoderHead(n_in=data_shape, n_out=n_out, **kwargs)


def make_decoder_head(data_shape, n_in, n_out_per_dim, **kwargs):
    return DecoderHead(n_in=n_in, n_out=(data_shape * n_out_per_dim), **kwargs)
