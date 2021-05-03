import os
from utils import make_atari
from gym.wrappers import Monitor
import numpy as np


def evaluate(args, model, step, cal_std=False):
    env = make_atari(args.env, skip=args.skip, max_episode_steps=args.max_moves)
    env = Monitor(env, directory=os.path.join(args.res_dir, str(step)), force=True)
    model.eval()
    done = True
    reward_lst = []
    for _ in range(args.evaluation_episodes):
        while True:
            if done:
                state, reward_sum, done = env.reset(), 0, False

            action = model.select_action(state, args.epsilon).item()  # Choose an action ε-greedily
            state, reward, done, _ = env.step(action)  # Step
            reward_sum += reward
            if args.render:
                env.render()

            if done:
                reward_lst.append(reward_sum)
                break
    env.close()
    reward_lst = np.array(reward_lst)
    if not cal_std:
        return reward_lst.mean()
    else:
        return reward_lst.mean(), reward_lst.std()