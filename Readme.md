(KDD 2023) Internal Logical Induction for Pixel-Symbolic Reinforcement Learning

# Installation

Install OpenJDK for *python-weka-wrapper3* (Ubuntu for example)
```bash
sudo apt-get update
sudo apt-get install openjdk-11-jdk
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

Install Ubuntu package 

```
sudo apt install xvfb # for CarRacing Env
sudo apt install swig # for gym[box2d]==0.21
```


Install Python Package

```bash
conda create -n ILI python=3.10
conda activate ILI
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install tensorboard tensorboardX scipy
pip install gym==0.21 'gym[box2d]==0.21' 'gym[atari]==0.21' 'gym[accept-rom-license]==0.21' 'gym[other]==0.21'
pip install UtilsRL==0.4.8 matplotlib imageio python-weka-wrapper3==0.2.11 nni pyglet==1.5.27 autopep8
```

We use gym 0.21 instead of the latest gymnasium 0.26 because gymnasium is incompatible with python-weka-wrapper3

Install modified [`Flappy Bird`](https://github.com/Talendar/flappy-bird-gym) env (Changed the installation dependencies and state representation)

```bash
pip install -e envs/flappy-bird-gym
```

Install modified [`Slime Volleyball`](https://github.com/hardmaru/slimevolleygym) env (Changed the state representation)

```bash
pip install -e envs/slimevolleygym
```

Install modified [`Collect Health`](https://github.com/hardmaru/slimevolleygym) env (Changed the state representation)

```bash
pip install -e envs/Miniworld
```

## [Optional] Use gym Wrapper `RecordVideo`

Set `use_recordvideo =  True` in `./configs/config.py`.

```cmd
conda install -c conda-forge ffmpeg
```
and then run

`xvfb-run -a python main.py`

# Experiment

The available options for `xxx` are `FlappyBird`, `CarRacing-v1`, and `CollectHealth-v1`.

## Exp1
```bash
bash scripts/exp1/PLP.sh
```

## Exp2
```bash
bash scripts/exp2/SymbolicDQN/xxx.sh
bash scripts/exp2/PixelDQN/xxx.sh
bash scripts/exp2/MixedDQN/xxx.sh
bash scripts/exp2/ILIDQN/xxx.sh
```

Citation
```
@inproceedings{DBLP:conf/kdd/0003CZYZ023,
  author       = {Jiacheng Xu and Chao Chen and Fuxiang Zhang and Lei Yuan and Zongzhang Zhang and Yang Yu},
  title        = {Internal Logical Induction for Pixel-Symbolic Reinforcement Learning},
  booktitle    = {Proceedings of the 29th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, {KDD} 2023, Long Beach, CA, USA, August 6-10, 2023},
  pages        = {2825--2837},
  publisher    = {{ACM}},
  year         = {2023},
  doi          = {10.1145/3580305.3599393},
}
```