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
from pathos.multiprocessing import Pool
import matplotlib.pyplot as plt
import pickle
import re

def dict_to_object(dic):
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    return Struct(**dic)



def save_obj(obj, file_name):
    output_hal = open(file_name, 'wb')
    str = pickle.dumps(obj)
    output_hal.write(str)
    output_hal.close
    
def load_obj(file_name):    
    with open(file_name,'rb') as file:
        rq  = pickle.loads(file.read())
    return rq






def test(input_args):    

    gail_experts_dir = 'data/generate_pred_data'


    RESULT_DIR = os.path.join('data', '.'.join(__file__.split('.')[:-1]))
    os.makedirs(RESULT_DIR, exist_ok=True)


    seed, lr, entropy_coef,  value_loss_coef, model_name_device = input_args

    model_name = model_name_device[0]
    device = model_name_device[1]



    token_eps = re.split('_|\.', model_name)[-2]

    token = 'eval_pred_data'.format(token_eps)
    print('Params: {}'.format(token))
    test_path = os.path.join(RESULT_DIR, token)
    os.makedirs(test_path, exist_ok=True)

    writer = SummaryWriter(os.path.join(test_path,'runs/evaluate'))




    args = {
        'algo': 'ppo',
        'gail': True,
        'gail_experts_dir': gail_experts_dir,
        'gail_batch_size': 128, 
        'gail_epoch': 5,
        'lr': lr,
        'eps': 1e-5,
        'alpha': 0.99,
        'gamma': 0.99 ,
        'use_gae': True,
        'gae_lambda': 0.95,
        'entropy_coef': entropy_coef,
        'value_loss_coef': value_loss_coef,
        'max_grad_norm': 0.5,
        'seed': seed,
        'cuda_deterministic': False,
        'num_processes': 1,
        'num_steps': 2048,
        'ppo_epoch': 10,
        'num_mini_batch': 32,
        'clip_param': 0.2,
        'log_interval': 1, 
        'save_interval': 1,
        'eval_interval': 10, 

        'num_env_steps': 10000000,

        'env_name': "HalfCheetah-v2",
        'log_dir': '/tmp/gym/', 
        'save_dir': './trained_models/',
        'no_cuda': False,
        'use_proper_time_limits': True,
        'recurrent_policy': False,
        'use_linear_lr_decay': True,
        'model_name': model_name

    }



    args = dict_to_object(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    # device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)


        # file_name = os.path.join(
        #     args.gail_experts_dir, "trajs_{}.pt".format(
        #         args.env_name.split('-')[0].lower()))
        
        file_name = os.path.join(args.gail_experts_dir, args.model_name)


        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs, obs_raw = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()


    

    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    
    print('args.num_env_steps: {}, args.num_steps: {}, args.num_processes {}'\
                .format(args.num_env_steps, args.num_steps, args.num_processes))


    mean_rewards = []

    num_updates = 1500
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, obs_raw, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            # if j >= 100:
            #     envs.venv.eval()

            gail_epoch = args.gail_epoch
            # if j < 10:
            if j < 100:
                gail_epoch = 100  # Warm up

            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()




        if (j % args.save_interval == 0):
            torch.save(agent.actor_critic.state_dict(), os.path.join(test_path, 'model_params_{}.pkl'.format(j)))
       
            save_obj(envs.venv.ob_rms, os.path.join(test_path, 'ob_rms_{}.pkl'.format(j)))


        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

       
        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            # ob_rms = utils.get_vec_normalize(envs).ob_rms
            
            ob_rms_load = load_obj(os.path.join(test_path, 'ob_rms_{}.pkl'.format(j)))
            eval_reward = evaluate(actor_critic, ob_rms_load, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


             # print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
             #                 len(eval_episode_rewards), np.mean(eval_episode_rewards)))
            print('Updates: {}, model_name: epoch_{}, mean_rewards during 50 trajs:{}'.format(j, token_eps, eval_reward))

            writer.add_scalar('mean_reward', eval_reward, global_step=j)
            mean_rewards.append(eval_reward)

    # plot
    mean_rewards_np = np.array(mean_rewards)
    plt.figure()
    plt.title('Epoch_{}'.format(token_eps))
    plt.plot([i*10+1 for i in range(len(mean_rewards))], mean_rewards_np)
    plt.grid()
    plt.savefig(os.path.join(test_path, 'test_reward.pdf'))
    plt.close()
    np.savetxt(os.path.join(test_path, 'test_reward.csv'), mean_rewards_np, delimiter=',')

if __name__ == "__main__":
    

    gail_experts_dir = 'cyy_result/cyy_scripts_20201107_gail_evaluate_I/generate_pred_dist_250_300_test_dataset_lr_3e-2_shuffle'
    devices = [torch.device("cuda:{}".format(i)) for i in [6,7]]
    model_names = ['pred_y_epoch_200.pt', 'pred_y_epoch_250.pt']
    model_names_devices = [(model_name, device) for model_name, device in zip(model_names, devices[:len(model_names)])]

    seed = 16
    lr = 3e-4
    entropy_coefs = 0
    value_loss_coefs = 0.5
    input_args = [seed, lr, entropy_coef, value_loss_coef, model_name_device[0]]

    test(input_args)





