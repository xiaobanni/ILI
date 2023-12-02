import torch
import torch.nn.functional as F
import gym
import torch as th
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """
        Multilayer Perceptron
        :param input_dim:
        :param output_dim:
        :param hidden_dim:
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """

        :param x: Input Layer
        :return: Output Layer
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.
    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim


class CNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, action_dim, features_dim=512, hidden_dim=128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32,
                      kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(
                observation_space.sample()[None]).float()).shape[1]

        self.flatten_linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.output_linear = MLP(
            features_dim, action_dim, hidden_dim=hidden_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.output_linear(self.flatten_linear(self.cnn(observations)))


class CNN_with_symbolic(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, symbolic_dim, action_dim, features_dim=512, hidden_dim=128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32,
                      kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(
                observation_space.sample()[None]).float()).shape[1]

        self.flatten_linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.output_linear = MLP(
            features_dim+symbolic_dim, action_dim, hidden_dim=hidden_dim)

    def forward(self, observations: th.Tensor, symbolic: th.Tensor) -> th.Tensor:
        if len(symbolic.shape) == 1:
            temp = th.cat([self.flatten_linear(self.cnn(observations)),
                           torch.Tensor(symbolic).cuda().unsqueeze(0)], dim=1)
        else:
            temp = th.cat([self.flatten_linear(self.cnn(observations)),
                           torch.Tensor(symbolic).cuda()], dim=1)
        return self.output_linear(temp)


class CNN_with_RL(BaseFeaturesExtractor):
    """ Add Representation Learning Loss with CNN """

    def __init__(self, observation_space: gym.spaces.Box, action_dim, predict_dim, features_dim=512, hidden_dim=128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32,
                      kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(
                observation_space.sample()[None]).float()).shape[1]

        self.flatten_linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.output_linear = MLP(
            features_dim, action_dim, hidden_dim=hidden_dim)

        # Add a prediction network for frame_state_batch
        self.prediction_network = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, predict_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        features = self.flatten_linear(self.cnn(observations))
        return self.output_linear(features), self.prediction_network(features)


if __name__ == '__main__':
    mlp = MLP(input_dim=16, output_dim=4)
    print(mlp.parameters)
    cnn = CNN(observation_space=gym.spaces.Box(
        0, 255, [3, 288, 512]), action_dim=2)
    print(cnn.parameters)
    cnn = CNN_with_symbolic(observation_space=gym.spaces.Box(
        0, 255, [3, 288, 512]), symbolic_dim=16, action_dim=2)
    print(cnn.parameters)
    cnn = CNN_with_RL(observation_space=gym.spaces.Box(
        0, 255, [3, 288, 512]), action_dim=2, predict_dim=16)
    print(cnn.parameters)
