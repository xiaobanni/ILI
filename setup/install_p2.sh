#!/bin/zsh

# Install conda packages
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tensorboard tensorboardX scipy

# Install pip packages
pip install gym==0.21 'gym[box2d]==0.21' 'gym[atari]==0.21' 'gym[accept-rom-license]==0.21' 'gym[other]==0.21'
pip install UtilsRL==0.4.8 matplotlib imageio python-weka-wrapper3==0.2.11 nni pyglet==1.5.27 autopep8

# Install modified Flappy Bird env
pip install -e envs/flappy-bird-gym

# Install modified Slime Volleyball env
pip install -e envs/slimevolleyballgym

# Install modified Collect Health env
pip install -e envs/Miniworld

# [Optional] Install ffmpeg for gym Wrapper RecordVideo
conda install -c conda-forge ffmpeg