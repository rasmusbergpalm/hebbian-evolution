from typing import Iterable

import numpy as np
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


def compute_centered_ranks(x: Iterable[float]) -> Iterable[float]:
    def compute_ranks(x):
        """
        Returns ranks in [0, len(x))
        Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
        """
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    x = np.array(x)
    assert x.ndim == 1
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y.tolist()


def get_writers(name):
    tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
    revision = os.environ.get("REVISION") or "%s" % datetime.now()
    message = os.environ.get('MESSAGE')

    train_writer = SummaryWriter(tensorboard_dir + '/%s/%s/train/%s' % (name, revision, message))
    test_writer = SummaryWriter(tensorboard_dir + '/%s/%s/test/%s' % (name, revision, message))
    return train_writer, test_writer
