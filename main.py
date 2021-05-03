import argparse
import os
from train import train
from eval import evaluate
from model import LinearQ, DQN, DuelingDQN
from utils import make_atari
import torch


def make_dir(args):
    from time import strftime, localtime
    time_tag = strftime('%Y-%m-%d %H:%M:%S', localtime())
    arch = args.arch
    if args.double:
        arch = 'Double' + arch
    args.res_dir = os.path.join(args.res_dir, args.env, arch, time_tag)
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--env', type=str, default='EnduroNoFrameskip-v4', help='Name of the environment')
    # training
    parser.add_argument('--steps', type=int, default=5e6, help='total steps')
    parser.add_argument('--eval_steps', type=int, default=1e5, help='evaluation steps')
    parser.add_argument('--start_steps', type=int, default=2e4, help='start steps')
    parser.add_argument('--target_steps', type=int, default=1e4, help='evaluation steps')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--hidden', type=int, default=512, help='hidden size of DQN')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--lr', type=float, default=0.0000625, help='learning rate')
    parser.add_argument('--discount', type=float, default=0.99, help='discount rate')
    parser.add_argument('--epsilon', type=float, default=0.05, help='discount rate')
    parser.add_argument('--evaluation_episodes', type=int, default=20, help='evaluation episodes')
    parser.add_argument('--render', type=bool, default=False, help='render of test')
    parser.add_argument('--res_dir', type=str, default='results/', help='results dir')
    parser.add_argument('--double', type=bool, default=False, help='use double')
    parser.add_argument('--arch', type=str, default='DeepQ', help='architecture',
                        choices=['LinearQ', 'DeepQ', 'DuelingDeepQ'])
    parser.add_argument('--eval', type=bool, default=False, help='only evaluate')
    parser.add_argument('--model_path', type=str, default='None', help='only evaluate')

    # env
    parser.add_argument('--skip', type=int, default=4, help='frame skip')
    parser.add_argument('--stack', type=int, default=4, help='frame stack')
    parser.add_argument('--max_moves', type=int, default=108000, help='max moves')
    parser.add_argument('--reward_clip', type=int, default=1, help='reward clip')

    # replay
    parser.add_argument('--capacity', type=int, default=1e6, help='capacity of replay buffer')

    args = parser.parse_args()
    make_dir(args)

    if not args.eval:
        train(args)
    else:
        args.evaluation_episodes = 100
        env = make_atari(args.env, skip=args.skip, max_episode_steps=args.max_moves)
        action_space = env.action_space.n
        assert os.path.exists(args.model_path)
        if args.arch == 'LinearQ':
            model = LinearQ(stack=args.stack, action_space=action_space).to(args.device)
        elif args.arch == 'DeepQ':
            model = DQN(stack=args.stack, hidden=args.hidden, action_space=action_space).to(args.device)
        else:
            model = DuelingDQN(stack=args.stack, hidden=args.hidden, action_space=action_space).to(args.device)

        model.load_state_dict(torch.load(args.model_path))
        avg_reward, std_reward = evaluate(args, model, 'final', save_video=False, cal_std=True)
        print('Final reward average and std is: {:.4f}({:.3f})'.format(avg_reward, std_reward))
