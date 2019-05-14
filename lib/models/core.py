import importlib

import numpy

import chainer
from chainer.backend import cuda
import chainer.distributions as D
import chainer.functions as F
from chainer import reporter

from lib import distributions
from lib import functions
from lib import xp_functions


class AvgELBOLoss(chainer.Chain):
    """Loss function of VAE.

    The loss value is equal to ELBO (Evidence Lower Bound)
    multiplied by -1.

    Args:
        encoder (chainer.Chain): A neural network which outputs variational
            posterior distribution q(z|x) of a latent variable z given
            an observed variable x.
        decoder (chainer.Chain): A neural network which outputs conditional
            distribution p(x|z) of the observed variable x given
            the latent variable z.
        prior (chainer.Chain): A prior distribution over the latent variable z.
        beta (float): Usually this is 1.0. Can be changed to control the
            second term of ELBO bound, which works as regularization.
        k (int): Number of Monte Carlo samples used in encoded vector.
    """

    def __init__(self, encoder, decoder, prior, beta=1.0, k=1):
        super(AvgELBOLoss, self).__init__()
        self.beta = beta
        self.k = k

        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder
            self.prior = prior

    def __call__(self, x):
        q_z = self.encoder(x)
        z = q_z.sample(self.k)
        p_x = self.decoder(z, n_batch_axes=2)
        p_z = self.prior()

        reconstr = F.mean(p_x.log_prob(
            F.broadcast_to(x[None, :], (self.k,) + x.shape)))
        kl_penalty = F.mean(q_z.log_prob(z) - p_z.log_prob(z))
        loss = - (reconstr - self.beta * kl_penalty)
        reporter.report({'loss': loss}, self)
        reporter.report({'reconstr': reconstr}, self)
        reporter.report({'kl_penalty': kl_penalty}, self)
        reporter.report({'beta': self.beta}, self)
        return loss

    def get_elbo(self, x, k=None, with_ll=False):
        if not k:
            k = self.k
        q_z = self.encoder(x)
        z = q_z.sample(k)
        p_x = self.decoder(z, n_batch_axes=2)
        p_z = self.prior()

        reconstr = p_x.log_prob(F.broadcast_to(x[None, :], (k,) + x.shape))
        kl_penalty = q_z.log_prob(z) - p_z.log_prob(z)

        elbo_k = reconstr - kl_penalty
        elbo = F.mean(elbo_k)
        if with_ll:
            log_likelihood = F.mean(F.logsumexp(elbo_k, axis=0) - numpy.log(k))
            return elbo, log_likelihood
        else:
            return elbo


class PriorHead(chainer.Link):

    def __init__(self, n_latent):
        super(PriorHead, self).__init__()

        self.loc = numpy.zeros([1, n_latent], numpy.float32)
        self.scale = numpy.ones([1, n_latent], numpy.float32)
        self.register_persistent('loc')
        self.register_persistent('scale')

    def forward(self):
        return self.loc, self.scale


class Distributionize(chainer.Chain):

    def __init__(self, head, to_dist_fn, no_input=False):
        super(Distributionize, self).__init__()
        with self.init_scope():
            self.head = head
            self.to_dist_fn = to_dist_fn
        if no_input:
            self.forward = self.forward_with_no_input

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__, self.head.__repr__())

    def forward(self, x, **kwargs):
        h = self.head(x, **kwargs)
        return self.to_dist_fn(h)

    def forward_with_no_input(self, **kwargs):
        h = self.head(**kwargs)
        return self.to_dist_fn(h)


def _make_head_fn(tower_type, module_name):
    module = importlib.import_module('lib.models.{}'.format(module_name))
    make_fn = getattr(module, 'make_{}_head'.format(tower_type))
    return make_fn


def make_encoder(model_type, dist_type, data_shape, **kwargs):
    to_dist_fn = None
    n_latent = kwargs['n_latent']

    if dist_type == 'euclid':
        n_out = n_latent * 2

        def to_dist_fn(h):
            return distributions.Independent(D.Normal(
                loc=h[..., :n_latent], scale=F.softplus(h[..., n_latent:])))
    elif dist_type == 'nagano':
        n_out = n_latent * 2

        def to_dist_fn(h):
            xp = cuda.get_array_module(h)
            scale = F.softplus(h[..., n_latent:])
            return distributions.HyperbolicWrapped(
                distributions.Independent(D.Normal(
                    loc=xp.zeros(shape=scale.shape, dtype=scale.dtype),
                    scale=scale)),
                functions.pseudo_polar_projection(h[..., :n_latent]))
    elif dist_type == 'nagano-unit':
        n_out = n_latent + 1

        def to_dist_fn(h):
            xp = cuda.get_array_module(h)
            scale = F.softplus(h[..., n_latent:])
            shape = scale.shape[:-1] + (n_latent,)
            return distributions.HyperbolicWrapped(
                distributions.Independent(D.Normal(
                    loc=xp.zeros(shape=shape, dtype=scale.dtype),
                    scale=scale)),
                functions.pseudo_polar_projection(h[..., :n_latent]))
    else:
        raise ValueError

    head = _make_head_fn('encoder', model_type)(
        data_shape=data_shape, n_out=n_out, **kwargs)
    return Distributionize(head, to_dist_fn)


def make_decoder(model_type, dist_type, data_shape, **kwargs):
    to_dist_fn = None
    n_latent = kwargs['n_latent']

    if kwargs['p_z'] == 'euclid':
        n_in = n_latent
    elif kwargs['p_z'] in ['nagano', 'nagano-unit']:
        n_in = n_latent + 1
    else:
        raise NotImplementedError

    if model_type == 'mlp':
        ndim = 1
    elif model_type == 'cnn':
        ndim = 3
    else:
        raise NotImplementedError

    if dist_type == 'normal':
        n_out_per_dim = 2

        def to_dist_fn(h):
            if ndim == 1:
                dim_h = h.shape[-1]
                loc = h[..., :(dim_h // 2)]
                base_sigma = h[..., (dim_h // 2):]
            elif ndim == 3:
                nb_channel = h.shape[-3]
                loc = h[..., :(nb_channel // 2), :, :]
                base_sigma = h[..., (nb_channel // 2):, :, :]
            else:
                raise NotImplementedError
            base_sigma += xp_functions._softplus_inverse(1.0)
            return distributions.Independent(
                D.Normal(
                    loc=loc,
                    scale=F.softplus(functions.clamp(
                        base_sigma, xp_functions._softplus_inverse(0.001)))),
                reinterpreted_batch_ndims=ndim)
    elif dist_type == 'bernoulli':
        n_out_per_dim = 1

        def to_dist_fn(h):
            return distributions.Independent(
                D.Bernoulli(logit=h), reinterpreted_batch_ndims=ndim)
    else:
        raise ValueError

    head = _make_head_fn('decoder', model_type)(
        data_shape=data_shape, n_in=n_in,
        n_out_per_dim=n_out_per_dim, **kwargs)
    return Distributionize(head, to_dist_fn)


def make_prior(model_type, dist_type, **kwargs):
    head = PriorHead(n_latent=kwargs['n_latent'])
    to_dist_fn = None

    if dist_type == 'euclid':
        def to_dist_fn(h):
            loc, scale = h
            return distributions.Independent(
                D.Normal(loc, scale), reinterpreted_batch_ndims=1)
    elif dist_type in ['nagano', 'nagano-unit']:
        def to_dist_fn(h):
            loc, scale = h
            xp = cuda.get_array_module(loc, scale)
            return distributions.HyperbolicWrapped(
                distributions.Independent(D.Normal(
                    loc=xp.zeros(shape=scale.shape, dtype=scale.dtype),
                    scale=scale)),
                functions.pseudo_polar_projection(loc))
    else:
        raise ValueError

    return Distributionize(head, to_dist_fn, no_input=True)
