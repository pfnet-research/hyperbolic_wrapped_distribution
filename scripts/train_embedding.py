import pathlib

import pandas as pd

import chainer
from chainer import training
from chainer.training import extensions

from lib import dataset
from lib.models import embedding
from lib.dataset import wordnet

from lib.extensions import Burnin
from lib.miscs import pathorganizer
from lib.miscs.hyperparams import Hyperparams
from lib.miscs.utils import assetsdir, datadir

from lib.miscs.logutils import logger_manager
logger = logger_manager.register(__name__)


def evaluate(hpt, train, test, loss):
    # calculate metric
    rank, mAP = wordnet.calculate_metrics(
        train, loss, k=100, verbose=True, gpu_mode=hpt.general.gpu >= 0)
    return {
        'rank': rank,
        'mAP': mAP
    }


def get_model(hpt):

    assert hpt.model.type == 'embedding'

    data_shape = dataset.get_data_shape(hpt.dataset.type, **hpt.dataset)
    encoder = embedding.make_encoder(
        n_in=data_shape,
        dist_type=hpt.model.p_z, **hpt.model)
    loss = embedding.EmbeddingLoss(encoder, hpt.loss.k, hpt.loss.bound)
    return loss


def main(hpt):

    logger.info('load dataset')
    train, valid, test = dataset.get_dataset(hpt.dataset.type, **hpt.dataset)
    assert valid is None
    assert test is None

    if hpt.general.test:
        train, _ = chainer.datasets.split_dataset(train, 100)
        chainer.set_debug(True)

    train_iter = chainer.iterators.SerialIterator(
        train, hpt.training.batch_size)

    logger.info('build model')
    loss = get_model(hpt)
    if hpt.general.gpu >= 0:
        loss.to_gpu(hpt.general.gpu)

    logger.info('setup optimizer')
    if hpt.optimizer.type == 'adam':
        optimizer = chainer.optimizers.Adam(alpha=hpt.optimizer.lr)
    elif hpt.optimizer.type == 'adagrad':
        optimizer = chainer.optimizers.AdaGrad(lr=hpt.optimizer.lr)
    else:
        raise AttributeError
    optimizer.setup(loss)

    logger.info('setup updater/trainer')
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer,
        device=hpt.general.gpu, loss_func=loss)

    trainer = training.Trainer(
        updater, (hpt.training.iteration, 'iteration'),
        out=po.namedir(output='str'))

    lr_name = 'alpha' if hpt.optimizer.type == 'adam' else 'lr'
    trainer.extend(Burnin(
        lr_name, burnin_step=hpt.training.burnin_step, c=hpt.training.c))

    trainer.extend(extensions.FailOnNonNumber())

    trainer.extend(
        extensions.snapshot_object(
            loss, 'loss_snapshot_iter_{.updater.iteration}'),
        trigger=(int(hpt.training.iteration / 5), 'iteration'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(
        extensions.PrintReport([
            'epoch', 'iteration', 'main/loss',
            'main/kl_target', 'main/kl_negative',
            'lr', 'main/bound', 'elapsed_time'
        ]))
    trainer.extend(extensions.ProgressBar())

    # Save plot images to the result dir
    if (not hpt.general.noplot) and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss'], 'epoch',
                file_name=(po.imagesdir() / 'loss.png').as_posix()))
        trainer.extend(
            extensions.PlotReport(
                ['main/kl_target', 'main/kl_negative'], 'epoch',
                file_name=(po.imagesdir() / 'kldiv.png').as_posix()))

    # Run the training
    logger.info('run training')
    trainer.run()

    logger.info('evaluate')
    metrics = evaluate(hpt, train, test, loss)
    for metric_name, metric in metrics.items():
        logger.info('{}: {:.4f}'.format(metric_name, metric))

    if hpt.general.noplot:
        return metrics

    return metrics


if __name__ == '__main__':
    current_file = pathlib.Path(__file__)
    hpt = Hyperparams(
        current_file.parents[1] / 'recipes/embedding_wordnet.yml')
    hpt = hpt.parse_args(program_name=current_file.name)

    logger.info(hpt.summary())

    po = pathorganizer.PathOrganizer(
        root=assetsdir, name=hpt.general.name, datadir=datadir)

    with (po.namedir() / 'option.yml').open('w') as f:
        hpt.dump(f)

    if hpt.general.num_experiments == 1:
        main(hpt)
    else:
        metrics = []
        for experiment_idx in range(hpt.general.num_experiments):
            logger.info('Trial [{}/{}] start'.format(
                experiment_idx + 1, hpt.general.num_experiments))
            if experiment_idx > 0:
                hpt['general']['noplot'] = True
            metrics.append(main(hpt))
        metrics = pd.DataFrame(metrics)
        logger.info('Total result:')
        for name in metrics.columns:
            logger.info('{}: {:.4f} \\pm {:.4f}'.format(
                name, metrics[name].mean(), metrics[name].std()))
        metrics.to_csv((po.logsdir() / 'metrics.tsv').as_posix(), sep='\t')
