import numpy as np

import chainer
from chainer.backends import cuda
import chainer.distributions as D
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import reporter

from lib import functions
from lib import distributions
from lib.models.core import Distributionize


class EmbeddingHead(chainer.Chain):

    def __init__(
            self, n_in, n_out, n_sigma,
            sigma_min=0.7, sigma_max=1.5, mu_max=2.0,
            initial_scale=0.01, **kwargs):

        super(EmbeddingHead, self).__init__()
        self.n_out = n_out
        self.n_sigma = n_sigma
        self.ln_sigma_min = np.log(sigma_min)
        self.ln_sigma_max = np.log(sigma_max)
        self.mu_max = mu_max

        with self.init_scope():
            self.mu = L.EmbedID(
                n_in, n_out, initialW=I.Normal(scale=initial_scale))
            ln_sigma_init_loc = (
                self.ln_sigma_max - self.ln_sigma_min
            ) / 2. + self.ln_sigma_min
            self.ln_sigma = L.EmbedID(
                n_in, n_sigma, initialW=np.float32(ln_sigma_init_loc))

    def forward(self, x):
        mu = self.mu(x)
        ln_sigma = self.ln_sigma(x)
        ln_sigma = F.broadcast_to(ln_sigma, mu.shape[:-1] + (self.n_out,))

        mu_norm = F.sqrt(F.sum(mu ** 2, axis=-1, keepdims=True))
        mu = mu / mu_norm * (- functions.clamp(-mu_norm, -self.mu_max))
        ln_sigma = F.clip(ln_sigma, self.ln_sigma_min, self.ln_sigma_max)

        return mu, ln_sigma


def make_encoder(n_in, dist_type, **kwargs):
    to_dist_fn = None
    n_latent = kwargs['n_latent']

    if 'euclid' in dist_type:

        def to_dist_fn(h):
            mu, ln_sigma = h
            return distributions.Independent(D.Normal(
                loc=mu, scale=F.softplus(ln_sigma)))

    elif 'nagano' in dist_type:

        def to_dist_fn(h):
            mu, ln_sigma = h
            xp = cuda.get_array_module(*h)
            scale = F.softplus(ln_sigma)
            return distributions.HyperbolicWrapped(
                distributions.Independent(D.Normal(
                    loc=xp.zeros(shape=scale.shape, dtype=scale.dtype),
                    scale=scale)),
                functions.pseudo_polar_projection(mu))
    else:
        raise ValueError

    n_sigma = 1 if 'unit' in dist_type else n_latent

    head = EmbeddingHead(n_in=n_in, n_out=n_latent, n_sigma=n_sigma, **kwargs)
    return Distributionize(head, to_dist_fn)


class EmbeddingLoss(chainer.Chain):

    def __init__(
            self, encoder, k=1, bound=0.1):
        super(EmbeddingLoss, self).__init__()

        self.k = k
        self.bound = bound

        with self.init_scope():
            self.encoder = encoder

    def forward(self, x):
        q_anchor = self.encoder(x[..., 0])
        q_target = self.encoder(x[..., 1])
        q_negative = self.encoder(x[..., 2])
        z = q_anchor.sample(self.k)

        logq_anchor = q_anchor.log_prob(z)
        kl_target = logq_anchor - q_target.log_prob(z)
        kl_negative = logq_anchor - q_negative.log_prob(z)

        energy = F.mean(F.relu(self.bound + kl_target - kl_negative))
        loss = energy

        reporter.report({'loss': loss}, self)
        reporter.report({'kl_target': F.mean(kl_target)}, self)
        reporter.report({'kl_negative': F.mean(kl_negative)}, self)
        reporter.report({'bound': self.bound}, self)

        return loss

    def calculate_energy(self, x, k=1000, batch_size=None):
        xp = cuda.get_array_module(x)
        kl_target = xp.zeros([len(x), 1], dtype='float32')
        if not batch_size:
            batch_size = len(x)
        nb_batch = np.ceil(len(x) / batch_size).astype(int)

        with chainer.using_config('train', False) and \
                chainer.no_backprop_mode():

            for i in range(nb_batch):
                idx_start = i * batch_size
                idx_end = (i + 1) * batch_size
                data = x[idx_start:idx_end]
                q_anchor = self.encoder(data[..., 0])
                q_target = self.encoder(data[..., 1])
                z = q_anchor.sample(k)

                kl_target[idx_start:idx_end, 0] = xp.mean(
                    q_anchor.log_prob(z).array - q_target.log_prob(z).array,
                    axis=0)

        return kl_target
