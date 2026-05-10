import copy
import json
import gym
import torch

from env.case_generator import CaseGenerator
from model.ternary_hgat_ppo import HGATPPO


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = json.load(open('config.json', 'r'))
    env_paras = cfg['env_paras']
    train_paras = cfg['train_paras']
    model_paras = cfg['model_paras']
    env_paras['device'] = device
    model_paras['device'] = device

    # small runnable instance
    env_paras['batch_size'] = 2
    env_paras['num_jobs'] = 10
    env_paras['num_mas'] = 5
    nums_ope = [5 for _ in range(env_paras['num_jobs'])]
    case = CaseGenerator(env_paras['num_jobs'], env_paras['num_mas'], 4, 6, nums_ope=nums_ope)
    env = gym.make('fjsp-v0', case=case, env_paras=env_paras)

    model = HGATPPO(model_paras, train_paras)
    state = env.state
    done = False
    steps = 0
    while not done and steps < 10:
        actions, _, _ = model.act(state)
        state, rewards, dones = env.step(actions)
        done = dones.all().item() if hasattr(dones, 'all') else bool(dones)
        steps += 1
    print('HGAT-PPO smoke run finished. steps=', steps)


if __name__ == '__main__':
    main()
