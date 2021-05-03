from tensorboardX import SummaryWriter
from utils import make_atari
from replay import ReplayMemory, Transition
from model import LinearQ, DQN, DuelingDQN

import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import os
from eval import evaluate
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import copy


def train(args):
    use_double_q = args.double
    summary_writer = SummaryWriter(log_dir=args.res_dir)

    env = make_atari(args.env, skip=args.skip, max_episode_steps=args.max_moves)
    memory = ReplayMemory(capacity=int(args.capacity))
    action_space = env.action_space.n

    if args.arch == 'LinearQ':
        online_net = LinearQ(stack=args.stack, action_space=action_space).to(args.device)
    elif args.arch == 'DeepQ':
        online_net = DQN(stack=args.stack, hidden=args.hidden, action_space=action_space).to(args.device)
    else:
        online_net = DuelingDQN(stack=args.stack, hidden=args.hidden, action_space=action_space).to(args.device)

    online_net.train()

    target_net = copy.deepcopy(online_net)

    optimizer = optim.Adam(online_net.parameters(), lr=args.lr, eps=1.5e-4)
    criterion = nn.SmoothL1Loss()

    done = True
    best_reward = -np.inf
    current_reward = -np.inf
    pbar = tqdm(range(int(args.steps)))
    for step in pbar:
        if done:
            loss_sum = []
            reward_sum = []
            state = env.reset()

        # random action for collecting initial data
        if step > args.start_steps:
            action = online_net.select_action(state, args.epsilon).item()
        else:
            action = random.randint(0, action_space - 1)
        next_state, reward, done, _ = env.step(action)
        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
        memory.push(state, action, next_state, reward, done)
        state = copy.deepcopy(next_state)
        reward_sum.append(reward)

        # train and test
        if step > args.start_steps and step % 4 == 0:
            transitions = memory.sample(args.batch_size)
            batch = Transition(*zip(*transitions))
            # prepare data
            state_batch = torch.from_numpy(np.asarray(batch.state)).to(args.device).float()
            action_batch = torch.from_numpy(np.asarray(batch.action)).to(args.device).unsqueeze(1).long()
            next_state_batch = torch.from_numpy(np.asarray(batch.next_state)).to(args.device).float()
            reward_batch = torch.from_numpy(np.asarray(batch.reward)).to(args.device).unsqueeze(1).float()
            final_mask = torch.tensor(batch.done, dtype=torch.bool).to(args.device)

            # Q value
            state_action_values = online_net(state_batch).gather(1, action_batch)

            if use_double_q:
                action_optim = online_net(next_state_batch).argmax(1, keepdim=True)
                next_state_values = target_net(next_state_batch).gather(1, action_optim).detach()
            else:
                next_state_values = target_net(next_state_batch).max(1)[0].unsqueeze(1).detach()
            next_state_values[final_mask] = 0.
            expected_state_action_values = (next_state_values * args.discount) + reward_batch

            # loss
            loss = criterion(state_action_values, expected_state_action_values)
            loss_sum.append(loss.item())
            summary_writer.add_scalar('loss', loss.item(), step)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(online_net.parameters(), 10)
            optimizer.step()

            if step % args.eval_steps == 0:
                online_net.eval()
                current_reward = evaluate(args, online_net, step)
                online_net.train()

                summary_writer.add_scalar('reward', current_reward, step)

                torch.save(online_net.state_dict(), os.path.join(args.res_dir, 'model_{}.pth'.format(step // args.eval_steps)))
                if best_reward < current_reward:
                    best_reward = current_reward
                    torch.save(online_net.state_dict(), os.path.join(args.res_dir, 'model.pth'))

            if step % args.target_steps == 0:
                target_net.load_state_dict(online_net.state_dict())
                target_net.eval()

            summary_writer.add_scalar('episode loss', np.array(loss_sum).mean(), step)
            summary_writer.add_scalar('episode reward', sum(reward_sum), step)
            pbar.set_description("Loss: %.8s, average episode reward %.6s, current/best test reward: %.6s(%.6s)" % (np.array(loss_sum).mean(), sum(reward_sum), current_reward, best_reward))
