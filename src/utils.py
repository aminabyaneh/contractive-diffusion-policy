"""
Utility functions for logging and wandb management.

This module provides functions to parse configuration files, create directories,
set random seeds, and manage logging with wandb. It includes a Logger class that
handles logging of metrics, model checkpoints, and video recordings during training
and evaluation.
"""

import random
import os
import uuid
import json
import wandb
import wandb.sdk.data_types.video as wv
import numpy as np
import torch
from omegaconf import OmegaConf

from cleandiffuser.env.wrapper import VideoRecordingWrapper
from datetime import datetime


def parse_cfg(cfg_path: str) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object."""
    base = OmegaConf.load(cfg_path)
    cli = OmegaConf.from_cli()
    for k,v in cli.items():
        if v == None:
            cli[k] = True
    base.merge_with(cli)
    return base


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Logger:
    """Primary logger object. Logs in wandb."""
    def __init__(self, log_dir, cfg):
        """
        Initializes the Logger object.
        This sets up the logging directories for metrics, models, and videos,
        and initializes wandb for logging

        Args:
            log_dir (str): The directory where logs will be stored.
            cfg (OmegaConf): Configuration object containing logging parameters.
        """
        self._log_dir = make_dir(log_dir)
        self._metrics_dir = make_dir(self._log_dir / 'metrics')
        self._model_dir = make_dir(self._log_dir / 'models')
        self._video_dir = make_dir(self._log_dir / 'videos')
        self._cfg = cfg

        # date and time based uuid
        self._uuid = f"{uuid.uuid4()}"

        # initialize wandb
        wandb.init(
            config=OmegaConf.to_container(cfg),
            project=cfg.project,
            group=cfg.group,
            name=cfg.exp_name,
            id=self._uuid,
            mode=cfg.wandb_mode,
            dir=self._log_dir
        )
        self._wandb = wandb

    def video_init(self, env, enable=False, video_id=""):
        """
        Initialize video recording for the environment.

        Args:
            env (VideoRecordingWrapper): The environment to record.
            enable (bool, optional): Whether to enable video recording. Defaults to False.
            video_id (str, optional): Identifier for the video file. Defaults to "".
        """
        # assert isinstance(env.env, VideoRecordingWrapper)
        if isinstance(env.env, VideoRecordingWrapper):
            video_env = env.env
        else:
            video_env = env

        if enable:
            video_env.video_recoder.stop()
            video_filename = os.path.join(self._video_dir, f"{video_id}_{wv.util.generate_id()}.mp4")
            video_env.file_path = str(video_filename)
        else:
            video_env.file_path = None

    def log(self, d: dict, category: str):
        """
        Log a dictionary of metrics to wandb and print them to the console.

        Args:
            d (dict): Dictionary containing metrics to log.
            category (str): Category of the metrics, either 'train' or 'eval'.
        """
        assert category in ['train', 'eval']
        assert 'step' in d

        print(f"[{d['step']}]", " | ".join(f"{k} {v:.6f}" for k, v in d.items() if k != 'step'))

        with (self._metrics_dir / f"{self._cfg.exp_name}.jsonl").open("a") as f:
            f.write(json.dumps({"step": d['step'], **d}) + "\n")
        _d = dict()

        for k, v in d.items():
            _d[category + "/" + k] = v

        self._wandb.log(_d, step=d['step'])

    def finish(self):
        """
        Finish the wandb run. This should be called at the end of training or evaluation.
        """
        if self._wandb:
            self._wandb.finish()
