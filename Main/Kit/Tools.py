# @Time     : 12/27/2021 2:09 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : Tools.py
# @Project  : QuantitativePrecipitationEstimation
import json
import logging
import os
import shutil

import torch


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file 'log_path'

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file.

    Example:
        logging.info("Starting training...")

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d: dict, json_path):
    """
    Saves dict of floats in json file

    Args:
        d: (dict) of float values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, "w") as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float)
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoints(state, checkpoint, saveName, is_best=False):
    """
    Saves model and training parameters at checkpoint + saveName + '.pth.tar'.
    if is_best==True, also saves checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        saveName: (string) saved file name
    """
    filePath = os.path.join(checkpoint, '{}.pth'.format(saveName))
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists!")
    torch.save(state, filePath)

    if is_best:
        shutil.copy(filePath, os.path.join(checkpoint, '{}_best.pth'.format(saveName)))


def load_checkpoint(checkpoint, model: torch.nn.Module, optimizer=None):
    """
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise IOError("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_dict"])

    return checkpoint
