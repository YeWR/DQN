import argparse
import os
from train import train, train_duel


def make_dir(args):
    from time import strftime, localtime
    time_tag = strftime('%Y-%m-%d %H:%M:%S', localtime())
    args.res_dir = os.path.join(args.res_dir, args.env, args.arch, time_tag)
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
    parser.add_argument('--evaluation_episodes', type=int, default=10, help='evaluation episodes')
    parser.add_argument('--render', type=bool, default=False, help='render of test')
    parser.add_argument('--res_dir', type=str, default='results/', help='results dir')
    parser.add_argument('--arch', type=str, default='DoubleDeepQ', help='architecture',
                        choices=['LinearQ', 'DoubleLinearQ', 'DeepQ', 'DoubleDeepQ', 'DuelingDeepQ'])

    # env
    parser.add_argument('--skip', type=int, default=4, help='frame skip')
    parser.add_argument('--stack', type=int, default=4, help='frame stack')
    parser.add_argument('--max_moves', type=int, default=108000, help='max moves')
    parser.add_argument('--reward_clip', type=int, default=1, help='reward clip')

    # replay
    parser.add_argument('--capacity', type=int, default=1e6, help='capacity of replay buffer')

    args = parser.parse_args()
    make_dir(args)

    if args.arch == 'LinearQ':
        train(args, use_cnn=False, use_double_q=False)
    elif args.arch == 'DoubleLinearQ':
        train(args, use_cnn=False, use_double_q=True)
    elif args.arch == 'DeepQ':
        train(args, use_cnn=True, use_double_q=False)
    elif args.arch == 'DoubleDeepQ':
        train(args, use_cnn=True, use_double_q=True)
    else:
        # DuelingDeepQ
        train_duel(args)
