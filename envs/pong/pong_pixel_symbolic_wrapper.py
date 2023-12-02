import gym
import matplotlib.pyplot as plt
import numpy as np


def process_frame(frame, normalize):
    board = np.where(frame[34:194, 140, 1] == 186)
    ball = np.where(frame[34:194, :, 1] == 236)
    if len(board[0]) == 0 or len(ball[0]) == 0 or len(ball[1]) == 0:
        return np.array([0, 0, 0])
    else:
        board_x = 140
        board_y = (np.max(board) + np.min(board)) / 2
        ball_x = (np.max(ball[1]) + np.min(ball[1])) / 2
        ball_y = (np.max(ball[0]) + np.min(ball[0])) / 2

    state = np.array([board_y, board_x - ball_x, board_y - ball_y])
    if normalize:
        state = state / 160

    return state


class PongPixelSymbolicWrapper(gym.Wrapper):
    def __init__(self, env, type):
        super(PongPixelSymbolicWrapper, self).__init__(env)

        self._normalize_obs = True
        self.type = type
        if self.type == "value":
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32)

        self.step_count = 0

    def reset(self, **kwargs):
        observation = super(PongPixelSymbolicWrapper, self).reset()
        value = process_frame(observation, self._normalize_obs)
        if self._normalize_obs:
            observation = observation / 255

        self.step_count = 0

        if self.type == "frame":
            return {"state": value, "pixels": observation}
        else:
            return value

    def step(self, action):
        self.step_count = self.step_count + 1
        observation, reward, done, info = super(PongPixelSymbolicWrapper, self).step(action)
        value = process_frame(observation, self._normalize_obs)
        if self._normalize_obs:
            observation = observation / 255

        # plt.imshow(observation)
        # plt.savefig("./figures/" + str(self.step_count) + "_" + str(value) + ".jpg")

        if self.type == "frame":
            return {"state": value, "pixels": observation}, reward, done, info
        else:
            return value, reward, done, info
