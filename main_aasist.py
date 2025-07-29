"""
Main script that evaluates fake or real audios using AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

'''the docker doesn't recongnize paths from system

so

docker run --gpus all -it -v /path/to/your/audio/folder:/container/audio/path <docker-image-name>
example: docker run --gpus all -it -v ./mydockerdata:/app/data deepfake_detector
'''

import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
from torch import Tensor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
import numpy as np

from aasist_utils import set_seed
import soundfile as sf
import librosa

warnings.filterwarnings("ignore", category=FutureWarning)

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def aasist_model(audio_path):
    
    # load experiment configurations
    
    config_file = 'config/AASIST.conf'
    
    with open(config_file, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    # optim_config = config["optim_config"]
    # optim_config["epochs"] = config["num_epochs"]
    # track = config["track"]
    # assert track in ["LA", "PA", "DF"], "Invalid track given"
    # if "eval_all_best" not in config:
    #     config["eval_all_best"] = "True"
    # if "freq_aug" not in config:
    #     config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(1234, config)
    
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("Device: {}".format(device))

    # define model architecture
    model = get_model(model_config, device)


    # evaluates pretrained model and exit script
    model.load_state_dict(
        torch.load(config["model_path"], map_location=device))
    # print("Model loaded : {}".format(config["model_path"]))

    # eval according to the choices by the user

    X,fs = sf.read(audio_path) 
    X_pad= pad(X,64600)
    x_inp= Tensor(X_pad)
    x_inp = x_inp.view(1, -1)
    # print("audio file loaded")

    # print("let's begin inference")
    model.eval()
    x_inp = x_inp.to(device)
    _,pred = model(x_inp)
    softmax_probs = torch.softmax(pred, dim=1)
    _, predicted_class = torch.max(softmax_probs, 1)
    spoofed_confidence_class_probs = softmax_probs[0,0]
    # Get the softmax probability values of the predicted class
    predicted_class_probs = softmax_probs.gather(1, predicted_class.unsqueeze(1)).squeeze(1)
        
    return spoofed_confidence_class_probs.item() , predicted_class.item()

    

def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    # print("no. model params:{}".format(nb_params))

    return model



