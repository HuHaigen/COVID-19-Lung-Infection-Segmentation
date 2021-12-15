import logging
import os
import shutil
import sys

import numpy as np
import torch
from resources.config import get_configs

config = get_configs()


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best validation accuracy so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            "Checkpoint directory does not exists. Creatding {}".format(checkpoint_dir))
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info("Saving last checkpoint")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info("Saving best checkpoint")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError("Checkpoint '{}' does not exist".format(checkpoint_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    state = torch.load(checkpoint_path, map_location="cuda:0")
    model.load_state_dict(state['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    return state


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


class RunningAcc:
    """Computes and stores the average
    """

    def __init__(self):
        self.A = 0
        self.B = 0
        self.I = 0
        self.mean_dice = 0
        self.iou = 0
        self.recall = 0
        self.pre = 0
        self.sum_dice = 0
        self.count = 0

    def update(self, A, B, I):
        self.A += A
        self.B += B
        self.I += I
        self.count += 1
        self.mean_dice += (2 * I + 1e-8) / (A + B + 1e-8)
        self.iou += (I + 1e-8) / (A + B - I + 1e-8)
        self.recall += (I + 1e-8) / (A + 1e-8)
        self.pre += (I + 1e-8) / (B + 1e-8)
        self.sum_dice = (2 * self.I + 1e-8) / (self.A + self.B + 1e-8)

        # if I==0:
        #     print(1)


def find_maximum_patch_size(model, device):
    """Tries to find the biggest patch size that can be send to GPU for inference
    without throwing CUDA out of memory"""
    logger = get_logger('PatchFinder')
    in_channels = model.in_channels

    patch_shapes = [(64, 128, 128), (96, 128, 128),
                    (64, 160, 160), (96, 160, 160),
                    (64, 192, 192), (96, 192, 192)]

    for shape in patch_shapes:
        # generate random patch of a given size
        patch = np.random.randn(*shape).astype('float32')

        patch = torch \
            .from_numpy(patch) \
            .view((1, in_channels) + patch.shape) \
            .to(device)

        logger.info("Current patch size: {shape}")
        model(patch)


def unpad(probs, index, shape, pad_width=8):
    def _new_slices(slicing, max_size):
        if slicing.start == 0:
            p_start = 0
            i_start = 0
        else:
            p_start = pad_width
            i_start = slicing.start + pad_width

        if slicing.stop == max_size:
            p_stop = None
            i_stop = max_size
        else:
            p_stop = -pad_width
            i_stop = slicing.stop - pad_width

        return slice(p_start, p_stop), slice(i_start, i_stop)

    D, H, W = shape

    i_c, i_z, i_y, i_x = index
    p_c = slice(0, probs.shape[0])

    p_z, i_z = _new_slices(i_z, D)
    p_y, i_y = _new_slices(i_y, H)
    p_x, i_x = _new_slices(i_x, W)

    probs_index = (p_c, p_z, p_y, p_x)
    index = (i_c, i_z, i_y, i_x)
    return probs[probs_index], index
