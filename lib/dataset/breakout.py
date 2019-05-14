import pathlib

import numpy as np
import h5py

import chainer

N_TRAIN = 90000
N_TEST = 10000


def load_dataset(path='~/data/breakout/state_samples/breakout_states_v2.h5',
                 withlabel=True):
    with h5py.File(pathlib.Path(path).expanduser().as_posix(), 'r') as hf:
        X = hf['states'][:].astype(np.float32).transpose(0, 3, 1, 2) / 255.
        X = X[..., 2:-2, 2:-2]
        X_train = X[:N_TRAIN]
        X_test = X[N_TRAIN:]
        if withlabel:
            t = hf['t'][:]
            try:
                r = hf['r'][:]
                cum_r = np.zeros(len(t) - 1)
                for start, end in zip(
                        np.arange(len(t))[t == 0],
                        np.r_[np.arange(len(t))[t == 0][1:], -1]):
                    cum_r[start:end] = r[start:end].cumsum()
                y_train = cum_r[:N_TRAIN]
                y_test = cum_r[N_TRAIN:]
            except KeyError:
                y_train = t[:N_TRAIN]
                y_test = t[N_TRAIN:]
    if withlabel:
        train = chainer.datasets.TupleDataset(X_train, y_train)
        test = chainer.datasets.TupleDataset(
            X_test[:len(y_test)], y_test)
        return train, test
    else:
        return X_train, X_test
