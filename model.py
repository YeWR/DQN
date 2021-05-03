import torch.nn as nn
import random
import torch


class QNetwork(nn.Module):
    def __init__(self, stack, action_space):
        super().__init__()
        self.stack = stack
        self.action_space = action_space

    def select_action(self, state, epsilon=0.05):
        if random.random() < epsilon:
            return torch.tensor([[random.randrange(self.action_space)]], dtype=torch.long)
        else:
            with torch.no_grad():
                return self.forward(torch.from_numpy(state).unsqueeze(0).cuda()).max(1)[1].view(1, 1)


class LinearQ(QNetwork):
    def __init__(self, stack, action_space):
        super().__init__(stack, action_space)
        self.input_size = 84 * 84 * stack
        self.fc = nn.Linear(self.input_size, action_space)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DQN(QNetwork):
    def __init__(self, stack, hidden, action_space):
        super().__init__(stack, action_space)

        self.convs = nn.Sequential(nn.Conv2d(stack, 32, 8, stride=4, padding=0), nn.ReLU(),
                                   nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())

        self.conv_output_size = 3136
        self.fc1 = nn.Linear(self.conv_output_size, hidden)
        self.fc2 = nn.Linear(hidden, action_space)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
