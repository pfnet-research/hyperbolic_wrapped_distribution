import chainer
import chainer.functions as F
import chainer.links as L


class EncoderHead(chainer.Chain):

    def __init__(self, data_shape, n_out, **kwargs):
        super(EncoderHead, self).__init__()

        self.n_out = n_out
        n_channel, width, height = data_shape

        if n_channel == 3:
            ch = 512
            # last_size = 4
        elif n_channel == 1:
            ch = 64
            # last_size = 10

        with self.init_scope():
            winit = chainer.initializers.Normal(0.02)
            self.c0_0 = L.Convolution2D(
                n_channel, ch // 4, 3, 1, 1, initialW=winit)
            self.c0_1 = L.Convolution2D(
                ch // 4, ch // 2, 4, 2, 1, initialW=winit)
            self.c1_0 = L.Convolution2D(
                ch // 2, ch // 2, 3, 1, 1, initialW=winit)
            self.c1_1 = L.Convolution2D(ch // 2, ch, 4, 2, 1, initialW=winit)
            self.c2_0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=winit)
            self.c2_1 = L.Convolution2D(ch, ch, 4, 2, 1, initialW=winit)

            self.bn0_1 = L.BatchNormalization(ch // 2, initial_gamma=0.1)
            self.bn1_0 = L.BatchNormalization(ch // 2, initial_gamma=0.1)
            self.bn1_1 = L.BatchNormalization(ch, initial_gamma=0.1)
            self.bn2_0 = L.BatchNormalization(ch, initial_gamma=0.1)
            self.bn2_1 = L.BatchNormalization(ch, initial_gamma=0.1)

            # self.out = L.Linear(
            #     last_size * last_size * ch, self.n_out, initialW=winit)
            self.out = L.Linear(self.n_out, initialW=winit)

    def forward(self, x):
        h = F.leaky_relu(self.c0_0(x))

        h = F.leaky_relu(self.bn0_1(self.c0_1(h)))
        h = F.leaky_relu(self.bn1_0(self.c1_0(h)))
        h = F.leaky_relu(self.bn1_1(self.c1_1(h)))
        h = F.leaky_relu(self.bn2_0(self.c2_0(h)))
        h = F.leaky_relu(self.bn2_1(self.c2_1(h)))

        h = F.reshape(h, (len(h), -1))

        out = self.out(h)
        return out


class DecoderHead(chainer.Chain):

    def __init__(self, data_shape, n_in, n_out_per_dim, **kwargs):
        super(DecoderHead, self).__init__()

        self.n_in = n_in
        n_channel, width, height = data_shape

        if n_channel == 3:
            ch = 512
            last_size = 4
        elif n_channel == 1:
            ch = 64
            last_size = 10
        self.ch = ch
        self.last_size = last_size
        self.nb_output_channel = n_out_per_dim * n_channel

        with self.init_scope():
            # winit = chainer.initializers.GlorotUniform()
            winit = chainer.initializers.Normal(0.02)

            self.l0 = L.Linear(
                self.n_in, last_size * last_size * ch, initialW=winit)

            self.c0_0 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=winit)
            self.c0_1 = L.Convolution2D(
                ch // 2, ch // 2, 3, 1, 1, initialW=winit)
            self.c1_0 = L.Deconvolution2D(
                ch // 2, ch // 4, 4, 2, 1, initialW=winit)
            self.c1_1 = L.Convolution2D(
                ch // 4, ch // 4, 3, 1, 1, initialW=winit)
            self.c2_0 = L.Deconvolution2D(
                ch // 4, self.nb_output_channel, 4, 2, 1, initialW=winit)
            self.c2_1 = L.Convolution2D(
                self.nb_output_channel, self.nb_output_channel,
                3, 1, 1, initialW=winit)

            self.bn0 = L.BatchNormalization(last_size * last_size * ch)
            self.bn0_0 = L.BatchNormalization(ch // 2)
            self.bn0_1 = L.BatchNormalization(ch // 2)
            self.bn1_0 = L.BatchNormalization(ch // 4)
            self.bn1_1 = L.BatchNormalization(ch // 4)

    def forward(self, z, n_batch_axes=1):
        assert n_batch_axes in (1, 2)
        if n_batch_axes == 2:
            sample_size = len(z)
            shp = (sample_size * z.shape[1],) + z.shape[2:]
            z = F.reshape(z, shp)

        batch_size = len(z)

        h = F.reshape(
            F.relu(self.bn0(self.l0(z))),
            (batch_size, self.ch, self.last_size, self.last_size))

        h = F.relu(self.bn0_0(self.c0_0(h)))
        h = F.relu(self.bn0_1(self.c0_1(h)))
        h = F.relu(self.bn1_0(self.c1_0(h)))
        h = F.relu(self.bn1_1(self.c1_1(h)))
        h = F.relu(self.c2_0(h))
        features = self.c2_1(h)
        if n_batch_axes == 2:
            features = F.reshape(
                features,
                (sample_size, batch_size // sample_size) + features.shape[1:])
        return features


def make_encoder_head(data_shape, n_out, **kwargs):
    return EncoderHead(data_shape=data_shape, n_out=n_out, **kwargs)


def make_decoder_head(data_shape, n_in, n_out_per_dim, **kwargs):
    return DecoderHead(
        data_shape=data_shape, n_in=n_in,
        n_out_per_dim=n_out_per_dim, **kwargs)
