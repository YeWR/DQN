import argparse
from utils import make_atari
from model import DQN
from replay import ReplayMemory, Transition

import os
import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


def test(args, model):
    env = make_atari(args.env, skip=args.skip, max_episode_steps=args.max_moves)
    model.eval()
    done = True
    avg_reward = []
    for _ in range(args.evaluation_episodes):
        while True:
            if done:
                state, reward_sum, done = env.reset(), 0, False

            action = model.select_action(state)  # Choose an action ε-greedily
            state, reward, done, _ = env.step(action)  # Step
            reward_sum += reward
            if args.render:
                env.render()

            if done:
                avg_reward.append(reward_sum)
                break
    env.close()
    avg_reward = sum(avg_reward) / len(avg_reward)
    return avg_reward


def main(args):
    env = make_atari(args.env, skip=args.skip, max_episode_steps=args.max_moves)
    memory = ReplayMemory(capacity=int(args.capacity))
    action_space = env.action_space.n

    policy_net = DQN(stack=args.stack, hidden=args.hidden, action_space=action_space).to(args.device)
    target_net = DQN(stack=args.stack, hidden=args.hidden, action_space=action_space).to(args.device)
    target_net.load_state_dict(policy_net.state_dict())
    policy_net.train()
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)

    done = True
    best_reward = -np.inf
    current_reward = -np.inf
    pbar = tqdm(range(int(args.steps)))
    for step in pbar:
        if done:
            state = env.reset()

        # random action for collecting initial data
        if step > args.start_steps:
            action = policy_net.select_action(state).item()
        else:
            action = random.randint(0, action_space - 1)
        next_state, reward, done, _ = env.step(action)
        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
        memory.push(state, action, next_state, reward, done)
        state = next_state

        # train and test
        if step > args.start_steps:
            transitions = memory.sample(args.batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.from_numpy(np.asarray(batch.state)).to(args.device).float()
            action_batch = torch.from_numpy(np.asarray(batch.action)).to(args.device).unsqueeze(1).long()
            next_action_batch = torch.from_numpy(np.asarray(batch.next_state)).to(args.device).float()
            reward_batch = torch.from_numpy(np.asarray(batch.reward)).to(args.device).unsqueeze(1).float()

            state_action_values = policy_net(state_batch).gather(1, action_batch)

            final_mask = torch.tensor(batch.done, dtype=torch.bool).to(args.device)
            next_state_values = target_net(next_action_batch).max(1)[0].detach()
            next_state_values[final_mask] = 0.
            expected_state_action_values = (next_state_values * args.discount) + reward_batch
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            if step % args.eval_steps == 0:
                policy_net.eval()
                current_reward = test(args, policy_net)
                policy_net.train()

                torch.save(policy_net.state_dict(), os.path.join(args.res_dir, 'model_{}.pth'.format(step // args.eval_steps)))
                if best_reward < current_reward:
                    best_reward = current_reward
                    torch.save(policy_net.state_dict(), os.path.join(args.res_dir, 'model.pth'))

            if step % args.target_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())
                target_net.eval()

            pbar.set_description("Loss: %.8s, current/best reward: %.6s(%.6s)" % (loss.item(), current_reward, best_reward))


def make_dir(args):
    from time import strftime, localtime
    time_tag = strftime('%Y-%m-%d %H:%M:%S', localtime())
    args.res_dir = os.path.join(args.res_dir, args.env, time_tag)
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--env', type=str, default='EnduroNoFrameskip-v4', help='Name of the environment')
    # training
    parser.add_argument('--steps', type=int, default=5e6, help='total steps')
    parser.add_argument('--eval_steps', type=int, default=1e5, help='evaluation steps')
    parser.add_argument('--start_steps', type=int, default=2e4, help='start steps')
    parser.add_argument('--target_steps', type=int, default=8e3, help='evaluation steps')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--hidden', type=int, default=512, help='hidden size of DQN')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--lr', type=float, default=0.00025, help='learning rate')
    parser.add_argument('--discount', type=float, default=0.99, help='discount rate')
    parser.add_argument('--evaluation_episodes', type=int, default=10, help='evaluation episodes')
    parser.add_argument('--render', type=bool, default=True, help='render of test')
    parser.add_argument('--res_dir', type=str, default='results/', help='results dir')


    # env
    parser.add_argument('--skip', type=int, default=4, help='frame skip')
    parser.add_argument('--stack', type=int, default=4, help='frame stack')
    parser.add_argument('--max_moves', type=int, default=108000, help='max moves')
    parser.add_argument('--reward_clip', type=int, default=1, help='reward clip')

    # replay
    parser.add_argument('--capacity', type=int, default=1e6, help='capacity of replay buffer')

    args = parser.parse_args()
    make_dir(args)
    main(args)
