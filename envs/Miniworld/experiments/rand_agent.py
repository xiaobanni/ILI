#!/usr/bin/env python3

import gym
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from utils import Flatten, init_weights, make_var


class Model(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            Flatten(),
            # Print(),
            nn.Linear(1120, 128),
            nn.LeakyReLU(),
        )

        self.rnn = nn.GRUCell(input_size=128, hidden_size=128)

        # GRU embedding to action
        self.action_probs = nn.Sequential(
            nn.Linear(128, num_actions), nn.LeakyReLU(), nn.LogSoftmax(dim=1)
        )

        self.apply(init_weights)

    def predict_action(self, img, memory):
        # batch_size = img.size(0)

        # x = img.view(batch_size, -1)
        x = self.encoder(img)

        memory = self.rnn(x, memory)
        action_probs = self.action_probs(memory)
        dist = Categorical(logits=action_probs)

        return dist, memory


##############################################################################

env = gym.make("MiniWorld-Hallway-v0")

print("num actions:", env.action_space.n)
print("max episode steps:", env.max_episode_steps)


def evaluate(model, seed=0, num_episodes=100):
    env = gym.make("MiniWorld-Hallway-v0")
    env.seed(seed)

    num_success = 0
    for i in range(num_episodes):
        # print(i)
        obs = env.reset()
        memory = Variable(torch.zeros([1, 128])).cuda()
        while True:
            obs = obs.transpose(2, 0, 1)
            obs = make_var(obs).unsqueeze(0)

            dist, memory = model.predict_action(obs, memory)
            action = dist.sample()

            obs, reward, done, info = env.step(action)
            if done:
                if reward > 0:
                    # print('success')
                    num_success += 1
                break

    return num_success / num_episodes


best_score = 0

for i in range(500):
    model = Model(env.action_space.n)
    model.cuda()

    s = evaluate(model)

    print(f"#{i + 1}: {s:.2f}")

    if s > best_score:
        best_score = s
        print(f"new best score: {s:.2f}")

    del model
    torch.cuda.empty_cache()

