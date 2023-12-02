from UtilsRL.misc import NameSpace
import torch


class Args(NameSpace):
    algo = "DQN"
    # MountainCar-v0, CartPole-v1, Acrobot-v1
    # FlappyBird-v0, FlappyBird-rgb-v0, CarRacing-v0, CarRacing-rgb-v0, CollectHealth-v0, CollectHealth-rgb-v0,
    env = "CollectHealth-rgb-v0"
    size = 100
    seed = 2022
    num_envs = 16  # parallel train envs
    num_eval_envs = 8  # parallel eval envs
    use_tensorboard = False
    eval_interval = 1_000  # Unit is number of steps
    eval_times = 1  # Eval times for each eval_interval, unit is number of num_eval_envs steps
    record_eval_video = True  # Record Eval Env Video
    use_nni = False  # if True, use nni to tune hyperparameters
    # NN (Neural Network Test Return) or Symbolic (Symbolic Return)
    nni_indicator = "NN"
    transfer = False

    # DQN
    gamma = 0.99
    epsilon_start = 1
    epsilon_end = 0.1
    epsilon_decay = 30_000
    lr = 3e-4
    capacity = 30_000  # Replay buffer capacity
    batch_size = 64
    learning_starts = 10_000  # Number of steps before learning starts
    train_freq = 16  # update the model every train_freq steps
    target_update = 100
    tau = 1
    feature_dim = 256
    hidden_dim = 128
    max_grad_norm = 10
    train_steps = 100_000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer_type = "ili"   # "simple", "ili", "mix", "per", "sil"
    load_network = False
    network_path = "/home/xiaox/code/symbolicRL/results/nips/CollectHealth/MixedDQN/s100_20230511-221139_mix_constant-0.5_[]_2023"

    # Use PLP policy to do exporation action
    dexp = False
    # Representation learning
    rl = False
    rl_coef = 0.1
    # Self-Imitation Learning
    sil = False
    sil_coef = 0.5

    # AuxiliarySignal
    # type of coefficient, "skew", "adaptation", "constant", "epsilon"
    intrinsic_type = "adaptation"
    # the decay constant used in "exponential" intrinsic_type, Former: ili_end_step
    ili_decay_constant = 50_000
    intrinsic_constant = 0.5  # constant intrinsic reward coefficient
    thresholds = 0.8  # when ripper_score * thresholds > RL_score , intrinsic reward works
    # maximum number of testing best symbolic, unit is number of num_eval_envs steps
    best_symbolic_test_num = 5 # size of the short sliding window
    window_size = 30  # size of the long sliding window
    skew_alpha = 5  # hyperparameter for skew distribution
    skew_dense = 0.001  # computing distribution intervals

    # PER
    alpha = 0.6
    beta = 0.1
    incremental_td_error = 1e-8

    # Ripper
    getPolicy = "topK"  # "topK", "lastN"
    # Determines the amount of data used for pruning. One fold is used for pruning, the rest for growing the rules.
    folds = 3  # Default: 3
    lastN = 2_000  # Use last N state-action pairs, Deprecated
    topK = 2_000  # Use trajectories with top K returns
    saveRuleInterval = 10  # Cooldown time for the saveIntervalRules function.
    load_ripper = False
    ripper_path = "/home/xiaox/code/symbolicRL/results/nips/CollectHealth/ILI/s100_20230514-022536_ili_constant-0.5_[500, 1000, 2000]_2023/ripper_data/CollectHealth-rgb-v0.118000.1.best.model"
