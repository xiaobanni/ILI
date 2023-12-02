from gym import utils
import gym
import miniworld
from miniworld.entity import MeshEnt
from miniworld.miniworld import MiniWorldEnv
import numpy as np
import matplotlib.pyplot as plt


def GetClockAngle(v1, v2):
    # 2个向量模的乘积
    TheNorm = np.linalg.norm(v1)*np.linalg.norm(v2)
    if np.isclose(TheNorm, 0):
        return 0
    # 叉乘
    rho = np.arcsin(np.cross(v1, v2)/TheNorm)
    # 点乘
    theta = np.arccos(np.dot(v1, v2)/TheNorm)
    if rho < 0:
        return - theta
    else:
        return theta


class CollectHealth(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Environment where the agent has to collect health kits and stay
    alive as long as possible. This is inspired from the VizDoom
    `HealthGathering` environment. Please note, however, that the rewards
    produced by this environment are not directly comparable to those
    of the VizDoom environment.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | move back                   |
    | 4   | pick up                     |
    | 5   | drop                        |
    | 6   | toggle / activate an object |
    | 7   | complete task               |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +2 for each time step
    -100 for dying

    ## Arguments

    ```python
    CollectHealth(size=16)
    ```

    `size`: size of the room

    """

    def __init__(self, type, size=100, place_range=3, symbolic_num=2, **kwargs):
        assert size >= 2
        self.size = size
        self.type = type
        self.place_range = place_range
        self.basic_health = place_range * 20

        self._normalize_obs = True
        self.symbolic_num = symbolic_num

        MiniWorldEnv.__init__(self, max_episode_steps=1000, **kwargs)
        utils.EzPickle.__init__(self, size, **kwargs)

    def _gen_world(self):
        # Create a long rectangular room
        self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex="cinder_blocks",
            floor_tex="slime",
        )

        # Place the agent a random distance away from the goal
        self.place_agent()

        for _ in range(1):
            self.box = self.place_entity(
                MeshEnt(mesh_name="medkit", height=0.8, static=False), place_range=self.place_range
            )

        self.health = self.basic_health

    def _calculate_symbolic(self):
        agent_pos = np.array(self.agent.pos)
        agent_dir = np.delete(self.agent.dir_vec, 1)
        medkit_pos = []
        for item in self.entities:
            if isinstance(item, miniworld.entity.MeshEnt):
                medkit_pos.append(item.pos)

        distances = [np.sqrt(sum((medkit_pos[i]-agent_pos)*(medkit_pos[i]-agent_pos)))
                     for i in range(len(medkit_pos))]
        target_index = np.argmin(distances)

        dis = distances[target_index]
        relative_dir = np.delete(medkit_pos[target_index] - agent_pos, 1)

        angle = GetClockAngle(agent_dir, relative_dir)

        # agent_pos = np.delete(agent_pos, 1)
        # wall_dis = [agent_pos[0]-0, self.size-agent_pos[0], agent_pos[1]-0, self.size-agent_pos[1]]
        # wall_dis = min(wall_dis)
        if self.symbolic_num == 2:
            return np.array([dis, angle])
        elif self.symbolic_num == 1:
            return np.array([angle])

    def reset(self):
        obs, _ = super(CollectHealth, self).reset()
        self.action_space = gym.spaces.Discrete(3)
        if self.type == "value":
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(2,), dtype=np.float32)
        value = self._calculate_symbolic()

        if self._normalize_obs:
            obs = obs / 255

        if self.type == "frame":
            return {"state": value, "pixels": obs}
        else:
            return value

    def step(self, action):
        obs, reward, termination, truncation, info = super(CollectHealth, self).step(action)
        value = self._calculate_symbolic()

        self.health -= 1

        if self.agent.carrying:
            self.entities.remove(self.agent.carrying)
            self.place_entity(self.agent.carrying, place_range=self.place_range)
            self.agent.carrying = None

            self.health = self.basic_health

        # # If the agent picked up a health kit
        # if action == self.actions.pickup:
        #     if self.agent.carrying:
        #         # Respawn the health kit
        #         self.entities.remove(self.agent.carrying)
        #         self.place_entity(self.agent.carrying)
        #         self.agent.carrying = None
        #
        #         # Reset the agent's health
        #         self.health = self.health + 20

        if self.health < 0:
            termination = True

        done = termination or truncation

        # Pass current health value in info dict
        info["health"] = self.health

        if self.type == "frame":
            return {"state": value, "pixels": obs}, reward, done, info
        else:
            return value, reward, done, info
