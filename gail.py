# Basic behavioural cloning
# Note: this uses gradient accumulation in batches of ones
#       to perform training.
#       This will fit inside even smaller GPUs (tested on 8GB one),
#       but is slow.
import os
from argparse import ArgumentParser
import pickle
import time
import copy
import gym
import minerl
import torch as th

import numpy as np
import wandb
import random, math
wandb.init(project="gail")
N_WORKERS = 30
from openai_vpt.agent import PI_HEAD_KWARGS, MineRLAgent
from data_loader import DataLoader
from openai_vpt.lib.tree_util import tree_map
import torch.nn.functional as F
K_epochs = 40

DEVICE = "cuda"
eps_clip = 0.2
mseratio=0.5
entropyratio=0.01
epochs=30
batchsize =40          # max timesteps in one episode
maxeps=10000
updateinterval=200
gamma=0.99
# Tuned with bit of trial and error
LEARNING_RATE = 0.000181
# OpenAI VPT BC weight decay
WEIGHT_DECAY = 0.039428
dgratio=10
MAX_GRAD_NORM = 5.0



def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def gail_train(name, data_dir, in_model, in_weights, out_weights):
    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=batchsize,
        n_epochs=epochs
    )
    wandb.config.batchsize=batchsize
    wandb.config.gamma=gamma
    wandb.config.K_epochs=K_epochs
    wandb.config.learning_rate=LEARNING_RATE
    wandb.config.eps_clip=eps_clip
    wandb.config.update_interval=updateinterval
    wandb.config.weight_decay=WEIGHT_DECAY
    wandb.config.entropy_ratio=entropyratio
    wandb.config.mse_ratio = mseratio
    wandb.config.dg_ratio=dgratio
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    device='cuda'
    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    env = gym.make(name)

    ppo = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    ppo.load_weights(in_weights)
    policy = ppo.policy
    # Freeze most params
    for param in policy.parameters():
        param.requires_grad = False
    # Unfreeze final layers
    G_trainable_parameters = []
    D_trainable_parameters = []
    # for param in policy.net.lastlayer.parameters():
    #     param.requires_grad = True
    #     G_trainable_parameters.append(param)
    #     D_trainable_parameters.append(param)
    for param in policy.pi_head.parameters():
        param.requires_grad = True
        G_trainable_parameters.append(param)
    for param in policy.dis_head.parameters():
        param.requires_grad = True
        D_trainable_parameters.append(param)

    # Parameters taken from the OpenAI VPT paper
    optimizer_G = th.optim.Adam(
        G_trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    optimizer_D = th.optim.Adam(
        D_trainable_parameters,
        lr=dgratio*LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    MseLoss = th.nn.MSELoss()

    episode_hidden_states = {}
    obs = env.reset()
    ppo.hidden_state = ppo.policy.initial_state(1)
    roll_out=[]
    batchnum=0

    for _ in range(1, maxeps + 1):
        hidden = ppo.hidden_state
        action = ppo.get_action(obs)

        action["ESC"] = 0
        img, reward, done, info = env.step(action)

        minerl_action = ppo._env_action_to_agent(action, to_torch=True, check_if_null=True)
        if minerl_action is None:
            # Action was null
            continue
        agent_obs = ppo._env_obs_to_agent(img)
        roll_out.append((agent_obs,minerl_action,hidden,done))

        if done:
            obs = env.reset()
            ppo.hidden_state = ppo.policy.initial_state(1)
        if len(roll_out)==updateinterval:

            tot_dloss=0.0
            for _ in range(K_epochs):
                diss=[]
                (batch_images, batch_actions, batch_episode_id) = data_loader.__next__()
                for image, action, episode_id in zip(batch_images, batch_actions, batch_episode_id):
                    agent_action = ppo._env_action_to_agent(action, to_torch=True, check_if_null=True)
                    if agent_action is None:
                        # Action was null
                        continue

                    agent_obs = ppo._env_obs_to_agent({"pov": image})

                    if episode_id not in episode_hidden_states:
                        # TODO need to clean up this hidden state after worker is done with the work item.
                        #      Leaks memory, but not tooooo much at these scales (will be a problem later).
                        episode_hidden_states[episode_id] = policy.initial_state(1)
                    agent_state = episode_hidden_states[episode_id]

                    _, _, d_prediction, new_agent_state = policy.get_output_for_observation(
                        agent_obs,
                        agent_state,
                        ppo._dummy_first
                    )
                    diss.append(d_prediction)
                    new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
                    episode_hidden_states[episode_id] = new_agent_state
                reals = th.squeeze(th.stack(diss, dim=0)).to(device)
                diss=[]
                fake_roll=random.sample(roll_out,batchsize)
                for (agent_obs,minerl_action,hidden,done) in fake_roll:
                    _, reward, _, _ = ppo.evaluate(agent_obs, minerl_action, hidden)
                    diss.append(reward)
                fakes = th.squeeze(th.stack(diss, dim=0)).to(device)

                d_loss = F.binary_cross_entropy_with_logits(reals, th.ones(reals.size()).to(
                    device)) + F.binary_cross_entropy_with_logits(fakes, th.zeros(fakes.size()).to(device))
                d_loss = d_loss.mean()
                optimizer_D.zero_grad()
                d_loss.backward()
                th.nn.utils.clip_grad_norm_(D_trainable_parameters, MAX_GRAD_NORM)
                optimizer_D.step()
                tot_dloss+=d_loss.detach()

            orewards=[]
            ologprobs=[]
            is_terminals=[]
            for (agent_obs, minerl_action, hidden, done) in roll_out:
                _, reward, log_prob, _ = ppo.evaluate(agent_obs, minerl_action, hidden)
                orewards.append(reward)
                ologprobs.append(log_prob)
                is_terminals.append(done)
            orewards = th.tensor(orewards, dtype=th.float32).to('cpu')
            with th.no_grad():
                orewards = th.sigmoid(orewards)
                orewards = orewards.log() - (1 - orewards).log()
            orewards = orewards.numpy().tolist()
            rewards=[]
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(orewards), reversed(is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            # Normalizing the rewards
            rewards = th.tensor(rewards, dtype=th.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            # convert list to tensor

            old_logprobs = th.squeeze(th.stack(ologprobs, dim=0)).to(device)
            tot_gloss = 0.0
            tot_entropy=0.0
            tot_mse=0.0
            tot_clip=0.0
            for _ in range(K_epochs):
                vs = []
                logprobs = []
                entropys = []
                for (agent_obs, minerl_action, hidden, done) in roll_out:
                    v, _, logprob, entropy = ppo.evaluate(agent_obs, minerl_action, hidden)
                    # print(minerl_action)
                    logprobs.append(logprob)
                    vs.append(v)
                    entropys.append(entropy)
                vs = th.squeeze(th.stack(vs, dim=0)).to(device)
                logprobs = th.squeeze(th.stack(logprobs, dim=0)).to(device)
                entropys = th.squeeze(th.stack(entropys, dim=0)).to(device)
                ratios = th.exp(logprobs - old_logprobs.detach())
                vs = th.squeeze(vs)

                # Finding Surrogate Loss
                advantages = rewards - vs.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                surr1 = ratios * advantages
                surr2 = th.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

                # final loss of clipped objective PPO
                clip_loss=-th.min(surr1, surr2)
                mse_loss= MseLoss(vs, rewards)
                g_loss = clip_loss + mseratio*mse_loss - entropyratio * entropys
                g_loss = g_loss.mean()
                optimizer_G.zero_grad()
                g_loss.backward()
                th.nn.utils.clip_grad_norm_(G_trainable_parameters, MAX_GRAD_NORM)
                optimizer_G.step()
                tot_gloss += g_loss.detach()
                tot_mse+=mse_loss.mean().detach()
                tot_clip+=clip_loss.mean().detach()
                tot_entropy+=entropys.mean().detach()
            # print(f"gloss: {tot_gloss:.4f} dloss: {tot_dloss:.4f}")
            wandb.log({'batch':batchnum,'gloss':tot_gloss,'dloss': tot_dloss,'cliploss': tot_clip,'mseloss': tot_mse,'entropy': tot_entropy})

            th.save(policy.state_dict(), out_weights + 'gail' + str(batchnum%10) + '.weights')
            roll_out = []
            # ppo.memory.clear()
            batchnum+=1











if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the directory containing recordings to be trained on")
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be finetuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where finetuned weights will be saved")

    args = parser.parse_args()
    gail_train(args.data_dir, args.in_model, args.in_weights, args.out_weights)
