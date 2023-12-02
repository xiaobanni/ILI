from utils.utils import get_env_name

BASELINE = {  # The lower bound of the return of the environment
    "MountainCar": -200,
    "CartPole": 0,
    "Acrobot": -500,
    "CarRacing": -120,
    "FlappyBird": 0,
    "CollectHealth": 0,
    "SlimeVolley": -3,
    "Pong": -21,
    "Breakout": 0,
}

# MountainCar-v0, CartPole-v1, Acrobot-v1
# FlappyBird-v0, FlappyBird-rgb-v0, CarRacing-v0, CarRacing-rgb-v0, SlimeVolley-v0, SlimeVolley-rgb-v0


def modify_config(cfg):
    if get_env_name(cfg.env) in ["MountainCar", "CartPole", "Acrobot"]:
        cfg.gamma = 0.99
        cfg.epsilon_start = 1
        cfg.epsilon_end = 0.01
        cfg.epsilon_decay = 30_000
        cfg.lr = 3e-4
        cfg.capacity = 30_000  # Replay buffer capacity
        cfg.batch_size = 64
        cfg.learning_starts = 0
        cfg.target_update = 100
        cfg.hidden_dim = 256
        cfg.train_steps = 200_000
        # Use trajectories with top K returns, Unit is number of steps
        cfg.topK = 2_000
        cfg.num_envs = 1
        cfg.train_freq = 1
        cfg.max_grad_norm = 0
    elif get_env_name(cfg.env) in ["FlappyBird", "CarRacing", "CollectHealth"]:
        cfg.eval_interval = 5_000
        cfg.eval_times = 1
        cfg.gamma = 0.99
        cfg.epsilon_start = 0.99
        cfg.epsilon_end = 0.01
        cfg.epsilon_decay = 100_000
        cfg.lr = 1e-4
        cfg.capacity = 50_000
        cfg.batch_size = 64
        cfg.learning_starts = 0
        cfg.target_update = 10_000
        cfg.feature_dim = 256
        cfg.hidden_dim = 128
        cfg.train_steps = 1_000_000
        cfg.best_symbolic_test_num = 5
        cfg.topK = 2_000
        if get_env_name(cfg.env) in ["CarRacing"]:
            cfg.train_steps = 500_000
        elif get_env_name(cfg.env) in ["CollectHealth"]:
            cfg.train_steps = 200_000
    return cfg
