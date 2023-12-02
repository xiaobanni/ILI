"""
Propositional Logic Policy
Propositional Logic Domain Knowledge
Internal Logic Induction
"""
import datetime
import os
import copy
import numpy as np
import nni
import gym
import weka.core.jvm as jvm
from UtilsRL.exp import parse_args
from torch.utils.tensorboard import SummaryWriter
from agents.dqn import DQNAgent
from symbolic.ripper import Ripper, Jrip
from configs.envSpecific import BASELINE, modify_config
from configs.nniUpdate import nni_update
from utils.nni_utils import NNI_Tool
from utils.eval import eval_mutlienv_policy
from utils.utils import get_env_name, get_env_information, get_dependencies_version, join_path, set_global_seeds, update_network, MovAvgSimple
from utils.logger import Logger
from utils.env_maker import singleenv_maker, multienv_maker
import warnings
warnings.filterwarnings("ignore")

curr_dir = os.path.dirname(__file__)
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

DEFAULT_TopK = {
    "CarRacing": [],
    "FlappyBird": [],
    "CollectHealth": [],
}


def get_dir_name(cfg):
    # prepare dircetory to store results and logs
    if cfg.intrinsic_type == "constant":
        intrinsic_type = "constant-" + str(cfg.intrinsic_constant)
    else:
        intrinsic_type = cfg.intrinsic_type
    dir_name = curr_time + "_" + cfg.buffer_type + "_" + \
        intrinsic_type+"_" + str(cfg.topK) + "_"+str(cfg.seed)
    if get_env_name(cfg.env) in ["CollectHealth"]:
        dir_name = "s"+str(cfg.size)+"_" + dir_name
    if cfg.dexp:
        dir_name = "dexp_"+dir_name
    if cfg.transfer:
        dir_name = "transfer_"+dir_name
    if cfg.load_network:
        dir_name = "load_"+dir_name
    if cfg.rl:
        dir_name = "rl_" + str(cfg.rl_coef) + "_" + dir_name
    return dir_name


def train(cfg, envs, agent, ripper):
    # prepare for training
    train_envs, eval_envs = envs
    total_reward = np.zeros(cfg.num_envs)
    # Initialize the baseline for logging before evaluation
    test_return = symbolic_return = BASELINE[get_env_name(cfg.env)]
    ripper_data_path = join_path(cfg.result_path, "ripper_data")
    state = train_envs.reset()
    if cfg.game_type == "frame":
        pixels, state = state["pixels"], state["state"]
    if cfg.transfer and cfg.load_ripper:  # SK DQN in Exp3
        jrip = Jrip()
        jrip.load_model(cfg.ripper_path)
        agent.best_ripper = {"ripper": jrip,
                             "return": MovAvgSimple(BASELINE[get_env_name(cfg.env)])}
        best_symbolic_return, _ = eval_mutlienv_policy(
            eval_envs, agent.best_ripper["ripper"], cfg, "ripper", i_step=0, eval_times=cfg.best_symbolic_test_num+1, record_eval_video=cfg.record_eval_video)
        agent.best_ripper["return"].add(
            best_symbolic_return, cfg.best_symbolic_test_num+1)
    else:
        agent.best_ripper = {"ripper": Jrip(),
                             "return": MovAvgSimple(BASELINE[get_env_name(cfg.env)])}
    nni_tool = NNI_Tool(cfg.use_nni, cfg.nni_indicator, cfg.train_steps//20)
    last_train_agent = last_update_target = last_eval = -1
    traj_storage = {idx: {'pixel_states': [], 'symbolic_states': [], 'actions': [
    ], 'rewards': [], 'next_pixels': [], 'dones': []} for idx in range(cfg.num_envs)}
    # start training
    print("=====Start training!=====")
    for i_steps in range(0, cfg.train_steps, cfg.num_envs):
        if cfg.game_type == "frame":  # pixel-symbolic env
            action = agent.choose_action(pixels, symbolic_state=state)
            next_state, reward, done, _ = train_envs.step(
                action[0])  # Use execution action, action[0]
            next_pixels, next_state = next_state["pixels"], next_state["state"]
            for idx in range(cfg.num_envs):
                if cfg.sil == True:
                    traj_storage[idx]['pixel_states'].append(pixels[idx])
                    traj_storage[idx]['symbolic_states'].append(state[idx])
                    traj_storage[idx]['actions'].append(action[0][idx])
                    traj_storage[idx]['rewards'].append(reward[idx])
                    traj_storage[idx]['next_pixels'].append(next_pixels[idx])
                    traj_storage[idx]['dones'].append(done[idx])
                elif cfg.buffer_type == "mix":
                    agent.replay_buffer.push(
                        pixels[idx], state[idx], action[0][idx], reward[idx], next_pixels[idx], next_state[idx], done[idx])
                else:
                    agent.replay_buffer.push(
                        pixels[idx], state[idx], action[0][idx], reward[idx], next_pixels[idx], done[idx])
            ripper_state = state.tolist()
            pixels, state = next_pixels, next_state
        else:  # vector env
            action = agent.choose_action(state, None)
            next_state, reward, done, _ = train_envs.step(
                action[0])  # Use execution action, action[0]
            for idx in range(cfg.num_envs):
                agent.replay_buffer.push(
                    None, state[idx], action[0][idx], reward[idx], next_state[idx], done[idx])
            ripper_state = state.tolist()
            state = next_state
        total_reward += reward
        ripper.add(ripper_state, action)

        for idx, d in enumerate(done):
            if d:
                if cfg.sil == True:
                    agent.replay_buffer.push(
                        traj_storage[idx]['pixel_states'],
                        traj_storage[idx]['symbolic_states'],
                        traj_storage[idx]['actions'],
                        traj_storage[idx]['rewards'],
                        traj_storage[idx]['next_pixels'],
                        traj_storage[idx]['dones']
                    )
                    traj_storage[idx] = {'pixel_states': [], 'symbolic_states': [
                    ], 'actions': [], 'rewards': [], 'next_pixels': [], 'dones': []}
                agent.logger.log("return/train_return",
                                 total_reward[idx], i_steps+idx)
                if cfg.getPolicy == "topK":
                    ripper.updateTopK(idx=idx, ret=total_reward[idx])
                ripper.train()
                for iidx, jrip in enumerate(ripper.jrip):
                    agent.logger.log(
                        f"ripper/rule_nums_{iidx}", jrip.rule_nums, i_steps+idx)
                print('Steps: {}/{}, Train Return:{:.1f}'.format(
                    i_steps + idx, cfg.train_steps, total_reward[idx]))
                total_reward[idx] = 0

        if (i_steps - last_train_agent) >= cfg.train_freq and i_steps >= cfg.learning_starts:
            last_train_agent = i_steps
            agent.update()
        if (i_steps - last_update_target) >= cfg.target_update:
            last_update_target = i_steps
            update_network(agent.q_value_net, agent.target_net, cfg.tau)
        if (i_steps - last_eval) >= cfg.eval_interval:
            eval_total_steps = 0
            last_eval = i_steps
            test_return, _ = eval_mutlienv_policy(
                eval_envs, agent, cfg, "dqn", i_step=i_steps, eval_times=cfg.eval_times, record_eval_video=cfg.record_eval_video)
            agent.test_return.add(test_return)
            agent.logger.log("return/test_return", test_return, i_steps)
            ripper_file_name = os.path.join(
                ripper_data_path, f"{cfg.env}.{i_steps}")
            ripper_update = False
            symbolic_return_max = BASELINE[get_env_name(cfg.env)]
            for idx, jrip in enumerate(ripper.jrip):  # ensemble
                if jrip.jrip_available == True:
                    symbolic_return, eval_steps = eval_mutlienv_policy(
                        eval_envs, jrip, cfg, "ripper", cfg.eval_times)
                    eval_total_steps += eval_steps
                    jrip.saveIntervalRules(
                        ripper_file_name+f".{idx}", symbolic_return)
                    if symbolic_return > agent.best_ripper["return"].mean:
                        ans, eval_steps = eval_mutlienv_policy(eval_envs, jrip,
                                                               cfg, "ripper", cfg.eval_times*2)
                        eval_total_steps += eval_steps
                        symbolic_return = (symbolic_return+ans*2)/3
                    # modify the best_ripper
                    if symbolic_return > agent.best_ripper["return"].mean:
                        agent.best_ripper["ripper"] = copy.deepcopy(jrip)
                        agent.best_ripper["return"].clear()
                        agent.best_ripper["return"].add(
                            symbolic_return, cfg.eval_times*3)
                        jrip.saveBestRules(
                            ripper_file_name+f".{idx}", symbolic_return)
                        ripper_update = True
                        ripper.best_jrip_idx = idx
                    # more test on best ripper
                    if agent.best_ripper["return"].n <= cfg.best_symbolic_test_num and agent.best_ripper["ripper"].jrip_available:
                        best_symbolic_return, eval_steps = eval_mutlienv_policy(
                            eval_envs, agent.best_ripper["ripper"], cfg, "ripper", i_step=i_steps, eval_times=cfg.eval_times, record_eval_video=cfg.record_eval_video)
                        eval_total_steps += eval_steps
                        agent.best_ripper["return"].add(
                            best_symbolic_return, cfg.eval_times)
                else:
                    symbolic_return = BASELINE[get_env_name(cfg.env)]
                symbolic_return_max = max(symbolic_return_max, symbolic_return)
                agent.logger.log(
                    f"ripper/return_{idx}", symbolic_return, i_steps)
            agent.logger.log("ripper/update", int(ripper_update), i_steps)
            agent.logger.log("ripper/best_idx",
                             ripper.best_jrip_idx, i_steps)
            agent.logger.log("return/symbolic_return_max",
                             symbolic_return_max, i_steps)
            agent.logger.log("return/best_symbolic_return",
                             agent.best_ripper["return"].mean, i_steps)
            nni_tool.report_intermediate_result(
                i_steps, symbolic_return, test_return)
            print('Steps: {}/{}, Test Return: {:.1f}, Symbolic Return Max: {:.1f}, Best Symbolic Return: {:.1f}'.format(
                i_steps, cfg.train_steps, test_return, symbolic_return_max, agent.best_ripper["return"].mean))
            i_steps += eval_total_steps
    nni_tool.report_final_result(symbolic_return, test_return)
    print("=====Finish training!=====")


def main():
    jvm.start()
    try:
        # get config
        cfg = parse_args(os.path.join("configs", "config.py")).Args
        cfg = modify_config(cfg)
        if isinstance(cfg.topK, (int, float)):
            cfg.topK = [cfg.topK]
        cfg.game_type = "frame" if "rgb" in cfg.env else "value"
        if cfg.use_nni == True:
            optimized_params = nni.get_next_parameter()
            cfg = nni_update(cfg, optimized_params)
        if cfg.buffer_type != "ili" and get_env_name(cfg.env) in DEFAULT_TopK:
            cfg.topK = DEFAULT_TopK[get_env_name(cfg.env)]
        dir_name = get_dir_name(cfg)
        cfg.result_path = join_path("results", cfg.env, dir_name)
        if cfg.record_eval_video == True:
            cfg.eval_video_path = join_path(cfg.result_path, "eval_videos")
        with open(os.path.join(cfg.result_path, "config.txt"), "w") as f:
            f.write(str(cfg))
        # prepare for training
        set_global_seeds(cfg.seed)
        get_dependencies_version()
        info_env = singleenv_maker(cfg.env, transfer=cfg.transfer)
        train_envs = multienv_maker(cfg.env, num_envs=cfg.num_envs,
                                    reward_shaping=True, transfer=cfg.transfer)
        eval_envs = multienv_maker(
            cfg.env, num_envs=cfg.num_eval_envs, transfer=cfg.transfer)
        get_env_information(env=info_env)
        state_space = info_env.observation_space
        symbolic_space = None
        if type(state_space) is gym.spaces.Dict:
            symbolic_space = state_space.spaces["state"]
            state_space = state_space.spaces["pixels"]
        action_space = info_env.action_space
        ripper = Ripper(cfg=cfg)
        logger = Logger()
        if cfg.use_tensorboard == True:
            logger.set_log(SummaryWriter(log_dir=cfg.result_path))
        if cfg.algo == "DQN":
            agent = DQNAgent(state_space, symbolic_space,
                             action_space, cfg, logger)
        else:
            raise NotImplementedError
        # start training
        train(cfg, [train_envs, eval_envs], agent, ripper)
        # save agent model
        agent.save(path=cfg.result_path)
        info_env.close()
        train_envs.close()
        eval_envs.close()
    finally:
        jvm.stop()


if __name__ == "__main__":
    main()
