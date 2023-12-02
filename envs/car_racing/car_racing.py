import gym
import matplotlib.pyplot as plt
import numpy as np
import cv2


def process_frame(frame, normalize, pre_direction=None, flag=False):
    row = frame[70, :, 1]
    line = frame[:, 47, 1]
    if row[47] != 0 and row[48] != 0:
        left_dis, right_dis, front_dis = 1, 1, 1
    else:
        left_side, right_side, front_side = 46, 49, 66
        while row[left_side] < 200 and left_side >= 1:
            left_side = left_side - 1
        while row[right_side] < 200 and right_side <= 94:
            right_side = right_side + 1
        while line[front_side] < 200 and front_side >= 1:
            front_side = front_side - 1
        left_dis = 46 - left_side
        right_dis = right_side - 49
        front_dis = 66 - front_side
        if normalize:
            left_dis, right_dis, front_dis = left_dis / 96, right_dis / 96, front_dis / 96

    if pre_direction is None:
        return np.array([left_dis, right_dis, front_dis])
    if not flag:
        return np.array([left_dis, right_dis, front_dis, pre_direction])

    frame = frame[:, :, 1]
    frame[np.where(frame <= 110)] = 100
    frame[66:77, 45:51] = 100

    if 0 < left_dis < 1 and 0 < right_dis < 1 and 0 < front_dis < 1:
        road = np.where(frame[80, 30:70] == 100)
        if len(road[0]) > 0:
            if np.min(road) > 0 and np.max(road) < 40 and np.max(road) - np.min(road) > 18:
                left_x = np.min(road) + 30
                right_x = np.max(road) + 30
                y = 80
                while y >= 0:
                    y = y - 5
                    left_to_calculate = frame[y:y+5, left_x - 5:left_x + 5]
                    right_to_calculate = frame[y:y+5, right_x-5:right_x+5]
                    if 255 in left_to_calculate:
                        direction = -1
                        break
                    if 255 in right_to_calculate:
                        direction = 1
                        break
                    left_temp = np.where(frame[y, left_x - 5:left_x + 6] == 100)
                    if len(left_temp[0]) > 0:
                        left_x = np.min(left_temp) + left_x - 5
                    else:
                        direction = pre_direction
                        break
                    right_temp = np.where(frame[y, right_x - 5:right_x + 6] == 100)
                    if len(right_temp[0]) > 0:
                        right_x = np.max(right_temp) + right_x - 5
                    else:
                        direction = pre_direction
                        break
                if y <= 0:
                    direction = 0
            else:
                direction = pre_direction
        else:
            direction = pre_direction
    else:
        direction = pre_direction

    return np.array([left_dis, right_dis, front_dis, direction])


class CarRacingPixelSymbolicWrapper(gym.Wrapper):
    def __init__(self, env, type):
        super(CarRacingPixelSymbolicWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(5)
        self.action_dict = {0: [0., 0., 0.], 1: [-1., 0., 0.],
                            2: [1., 0., 0.], 3: [0., 1., 0.], 4: [0., 0., 1.]}

        self._normalize_obs = True
        self.type = type
        if self.type == "value":
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.symbolic_num = 3

    def reset(self, **kwargs):
        observation = super(CarRacingPixelSymbolicWrapper, self).reset()
        value = process_frame(observation, self._normalize_obs)
        if self._normalize_obs:
            observation = observation / 255

        if self.type == "frame":
            return {"state": value, "pixels": observation}
        else:
            return value

    def step(self, action):
        action = self.action_dict[action]
        observation, reward, done, info = super(CarRacingPixelSymbolicWrapper, self).step(action)
        value = process_frame(observation, self._normalize_obs)
        if self._normalize_obs:
            observation = observation / 255

        if not (value > 0).all():
            done = True

        if self.type == "frame":
            return {"state": value, "pixels": observation}, reward, done, info
        else:
            return value, reward, done, info


class CarRacingPixelSymbolicWrapperWithDirection(gym.Wrapper):
    def __init__(self, env, type):
        super(CarRacingPixelSymbolicWrapperWithDirection, self).__init__(env)
        self.action_space = gym.spaces.Discrete(5)
        self.action_dict = {0: [0., 0., 0.], 1: [-1., 0., 0.],
                            2: [1., 0., 0.], 3: [0., 1., 0.], 4: [0., 0., 1.]}

        self._normalize_obs = True
        self.type = type
        if self.type == "value":
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.symbolic_num = 4

        self.pre_direction = 0
        self.step_count = 0
        self.calculate_direction_interval = 1

    def reset(self, **kwargs):
        self.pre_direction = 0
        self.step_count = 0

        observation = super(CarRacingPixelSymbolicWrapperWithDirection, self).reset()
        value = process_frame(observation, self._normalize_obs, self.pre_direction, True)
        self.pre_direction = value[-1]
        assert self.pre_direction in [0, -1, 1]
        if self._normalize_obs:
            observation = observation / 255

        if self.type == "frame":
            return {"state": value, "pixels": observation}
        else:
            return value

    def step(self, action):
        action = self.action_dict[action]
        observation, reward, done, info = super(CarRacingPixelSymbolicWrapperWithDirection, self).step(action)
        value = process_frame(observation, self._normalize_obs, self.pre_direction,
                              self.step_count % self.calculate_direction_interval == 0)
        self.pre_direction = value[-1]
        assert self.pre_direction in [0, -1, 1]

        # plt.imshow(observation)
        # plt.savefig("./figures/" + str(self.step_count) + "_" + str(self.pre_direction) + ".jpg")

        if self._normalize_obs:
            observation = observation / 255

        self.step_count = self.step_count + 1

        if not (value[:-1] > 0).all():
            done = True

        if self.type == "frame":
            return {"state": value, "pixels": observation}, reward, done, info
        else:
            return value, reward, done, info


class CarRacingPixelSymbolicWrapperWithDirandVel(gym.Wrapper):
    def __init__(self, env, type):
        super(CarRacingPixelSymbolicWrapperWithDirandVel, self).__init__(env)
        self.action_space = gym.spaces.Discrete(5)
        self.action_dict = {0: [0., 0., 0.], 1: [-1., 0., 0.],
                            2: [1., 0., 0.], 3: [0., 1., 0.], 4: [0., 0., 1.]}

        self._normalize_obs = True
        self.type = type
        if self.type == "value":
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.symbolic_num = 5

        self.pre_direction = 0
        self.step_count = 0
        self.calculate_direction_interval = 1
        self.velocity = None
        self.change_count = 0

    def reset(self, **kwargs):
        self.pre_direction = 0
        self.step_count = 0
        self.change_count = 0

        observation = super(CarRacingPixelSymbolicWrapperWithDirandVel, self).reset()
        value = process_frame(observation, self._normalize_obs, self.pre_direction, True)
        self.pre_direction = value[-1]
        assert self.pre_direction in [0, -1, 1]
        self.velocity = np.sqrt(np.sum(np.array(self.car.hull.linearVelocity) ** 2))
        value = np.append(value, self.velocity)
        if self._normalize_obs:
            observation = observation / 255

        if self.type == "frame":
            return {"state": value, "pixels": observation}
        else:
            return value

    def step(self, action):
        action = self.action_dict[action]
        observation, reward, done, info = super(CarRacingPixelSymbolicWrapperWithDirandVel, self).step(action)
        value = process_frame(observation, self._normalize_obs, self.pre_direction,
                              self.step_count % self.calculate_direction_interval == 0)
        if self.pre_direction != value[-1]:
            self.change_count = self.change_count + 1
        if self.change_count >= 3:
            self.change_count = 0
            self.pre_direction = value[-1]
        else:
            value[-1] = self.pre_direction

        self.velocity = np.sqrt(np.sum(np.array(self.car.hull.linearVelocity) ** 2))
        value = np.append(value, self.velocity)

        # plt.imshow(observation)
        # plt.savefig("./figures/" + str(self.step_count) + "_" + str(self.pre_direction) + ".jpg")

        if self._normalize_obs:
            observation = observation / 255

        self.step_count = self.step_count + 1

        if not (value[:-2] > 0).all():
            done = True

        if self.type == "frame":
            return {"state": value, "pixels": observation}, reward, done, info
        else:
            return value, reward, done, info