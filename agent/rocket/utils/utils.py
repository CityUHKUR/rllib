import os
import logging
import time
import torch
from functools import reduce

logger = logging.getLogger(__name__)


def has_checkpoint(checkpoint_dir):
    """
    True if a checkpoint is present in checkpoint_dir
    """
    if os.path.isdir(checkpoint_dir):
        if len(os.listdir(checkpoint_dir)) > 0:
            return os.path.isfile(os.path.join(checkpoint_dir, "checkpoint"))

    return False


def chain(input, layers):
    x = input
    for layer in layers:
        x = layer(x)
    return x


def isin(x, sets):
    return reduce(lambda l, r: l or r, map(lambda elem: x == elem, sets))


@torch.no_grad()
def apply_functions(layer, target_models, funcs):
    if isin(type(layer), target_models):
        for func in funcs:
            func(layer)


def wait_for_checkpoint(checkpoint_dir, data_store=None, retries=10):
    """
    block until there is a checkpoint in checkpoint_dir
    """
    for i in range(retries):
        if data_store:
            data_store.load_from_store()

        if has_checkpoint(checkpoint_dir):
            return
        time.sleep(10)

    raise ValueError((
        'Tried {retries} times, but checkpoint never found in '
        '{checkpoint_dir}'
    ).format(
        retries=retries,
        checkpoint_dir=checkpoint_dir,
    ))
