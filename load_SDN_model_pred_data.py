import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pickle
import re
from tools import *


def dict_to_object(dic):
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    return Struct(**dic)







if __name__ == '__main__':


    RESULT_DIR = os.path.join('data', '.'.join(__file__.split('.')[:-1]))
    os.makedirs(RESULT_DIR, exist_ok=True)
 
    token = 'generate_pred_data'
    test_path = os.path.join(RESULT_DIR, token)
    os.makedirs(test_path, exist_ok=True)



    
  
    path = 'data/trained_model_path'

    model_names = ['model_params_999.pkl']





    seed = 16 

    device_ids = [0,1,2,3,4,5,6,7]
    device = torch.device("cuda: 0")
    
    args = {

            'test_batchsize' : 10, 
            'epochs' : 1000,
            'npoints' : 1000,

            'd': 23, 
            'nsimu' : 500,
            'no_cuda' : False,
            'seed' : seed,
            'log_interval' : 10,
            'eps' : 0.01,
            'decay_rate': 1,
            'decay_step': 1000

        }


    args = dict_to_object(args)
    
   

    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    # load train dataset
    train_data_path = 'data/concat_train_test_data_for_SDN_training/train_test_xy.csv'
    train_xy = np.loadtxt(train_data_path, delimiter=',')
    train_x = train_xy[100:, 23000:]





    # load model
    model = Net(d=args.d, N=args.npoints)

    if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model = nn.DataParallel(model, device_ids= device_ids)
    
    model.to(device)


    for model_name in model_names:

        token_eps = re.split('_|\.', model_name)[-2]
        model.load_state_dict(torch.load(os.path.join(path,model_name)))

        # pred_y
        train_x = torch.Tensor(train_x).to(device)
        output = model(train_x*10)
        torch.cuda.empty_cache()
        # pred_y.shape = (50,1000,23)
        pred_y = (output.cpu().detach().reshape(output.shape[0], args.npoints, -1))/10


      
        pred_y_states = pred_y[:, :, :17]
        pred_y_actions = pred_y[:, :, 17:]
        pred_y_lengths = np.array([1000]*50)

        pred_y_dict = {}
        pred_y_dict['states'] = pred_y_states
        pred_y_dict['actions'] = pred_y_actions
        pred_y_dict['lengths'] = pred_y_lengths

        torch.save(pred_y_dict, os.path.join(test_path, 'pred_y_epoch_{}.pt'.format(token_eps)))





