from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym import ObservationWrapper
import cv2
from utils.utils import get_env_name


class LazyFrames(object):
    r"""Ensures common frames are only stored once to optimize memory use.
    To further reduce the memory use, it is optionally to turn on lz4 to
    compress the observations.
    .. note::
        This object should only be converted to numpy array just before forward pass.
    Args:
        lz4_compress (bool): use lz4 to compress the frames internally
    """
    __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames")

    def __init__(self, frames, lz4_compress=False):
        self.frame_shape = tuple(frames[0].shape)
        self.shape = (len(frames),) + self.frame_shape
        self.dtype = frames[0].dtype
        if lz4_compress:
            from lz4.block import compress

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            # single frame
            return self._check_decompress(self._frames[int_or_slice])
        return np.stack(
            [self._check_decompress(f) for f in self._frames[int_or_slice]], axis=0
        )

    def __eq__(self, other):
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress

            return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(
                self.frame_shape
            )
        return frame


class PreprocessingObservation(ObservationWrapper):
    """
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
    """

    def __init__(self, env, num_stack=4, lz4_compress=False, transfer=False):
        super().__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress
        self.transfer = transfer
        self.frames = deque(maxlen=num_stack)
        self.observation_space = spaces.Dict()
        self.observation_space.spaces["pixels"] = spaces.Box(
            low=0.0, high=1.0, shape=(4, 84, 84), dtype=np.float32)

        if get_env_name(env.spec.id) == "FlappyBird":
            self.observation_space.spaces["state"] = spaces.Box(
                -np.inf, np.inf, shape=(2, ), dtype=np.float32)
        elif get_env_name(env.spec.id) == "CarRacing":
            self.observation_space.spaces["state"] = spaces.Box(
                -np.inf, np.inf, shape=(env.symbolic_num, ), dtype=np.float32)
        elif get_env_name(env.spec.id) == "SlimeVolleyNoFrameskip":
            self.observation_space.spaces["state"] = spaces.Box(
                -np.inf, np.inf, shape=(4, ), dtype=np.float32)
        elif get_env_name(env.spec.id) == "Pong":
            self.observation_space.spaces["state"] = spaces.Box(
                -np.inf, np.inf, shape=(3, ), dtype=np.float32)
        elif get_env_name(env.spec.id) == "Breakout":
            self.observation_space.spaces["state"] = spaces.Box(
                -np.inf, np.inf, shape=(3, ), dtype=np.float32)
        elif get_env_name(env.spec.id) == "MiniWorld":
            self.observation_space.spaces["state"] = spaces.Box(
                -np.inf, np.inf, shape=(env.symbolic_num, ), dtype=np.float32)
        else:
            raise ValueError("Unknown env: {}".format(env.spec.id))

    def observation(self):
        assert len(self.frames) == self.num_stack, (len(
            self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.transfer:
            observation["pixels"] = self.transfer_frame(observation["pixels"])
        observation["pixels"] = self.resize_gray(observation["pixels"])
        self.frames.append(observation["pixels"])
        return {"state": observation["state"], "pixels": self.observation()}, reward, done, info

    def reset(self):
        observation = self.env.reset()
        if self.transfer:
            observation["pixels"] = self.transfer_frame(observation["pixels"])
        observation["pixels"] = self.resize_gray(observation["pixels"])
        [self.frames.append(observation["pixels"])
         for _ in range(self.num_stack)]
        return {"state": observation["state"], "pixels": self.observation()}

    def resize_gray(self, observation):
        observation = cv2.resize(
            observation, (84, 84), interpolation=cv2.INTER_AREA)
        observation = cv2.cvtColor(
            observation.astype(np.float32), cv2.COLOR_BGR2GRAY)
        return observation

    def transfer_frame(self, observation):
        if (observation <= 1.0).all():
            observation = 1 - observation
        else:
            observation = 255 - observation

        return observation
