import sys
import json
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
import librosa
from importlib import import_module
from typing import Dict, List, Union
from aasist_utils import set_seed


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x 

    
    
def rawnet_model(audio_path):
    
    config_file = 'config/RawNet.conf'

    with open(config_file, "r") as f_json:
        config = json.loads(f_json.read())
        
    model_config = config['model_config']
 
    # set_random_seed(1234)
    set_seed(1234, config)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # model 
    model = get_model(model_config, device)
    model =(model).to(device)
    
    
    model_path = 'models/weights/pre_trained_DF_RawNet2.pth'
    
    if model_path:
        model.load_state_dict(torch.load(model_path,map_location=device))
        # print('Model loaded : {}'.format(model_path))
        
    if audio_path:
        pass
        # print('Audio loaded : {}'.format(audio_path))
        
    X,fs = librosa.load(audio_path, sr=16000) 
    X_pad= pad(X,64600)
    x_inp= Tensor(X_pad)
    x_inp = x_inp.view(1, -1)
    # Perform a forward pass on a single audio file

    model.eval()
    x_inp = x_inp.to(device)
    pred = model(x_inp)
    softmax_probs = torch.softmax(pred, dim=1)
    _, predicted_class = torch.max(softmax_probs, 1)
    spoofed_confidence_class_probs = softmax_probs[0,0]
    # Get the softmax probability values of the predicted class
    predicted_class_probs = softmax_probs.gather(1, predicted_class.unsqueeze(1)).squeeze(1)
    
    return spoofed_confidence_class_probs.item(), predicted_class.item()



def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "RawNet")
    model = _model(model_config , device).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    # print("no. model params:{}".format(nb_params))

    return model
    