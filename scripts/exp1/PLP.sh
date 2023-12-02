#!/bin/bash

# ./script.sh -p 4 -g 0,1,2,3


# Set default values
parallel_seeds=2
available_gpus=(0)

# Parse command line arguments
while getopts ":p:g:" opt; do
  case $opt in
    p)
      parallel_seeds="$OPTARG"
      ;;
    g)
      IFS=',' read -ra available_gpus <<< "$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

log_dir="log/exp1_PLP/$(date +%Y%m%d-%H%M%S)"
if [ ! -d "$log_dir" ]; then
  echo "Creating log directory $log_dir"
  mkdir -p "$log_dir"
fi

envs=(MountainCar-v0 CartPole-v1 Acrobot-v1)
num_gpus=${#available_gpus[@]}

for e in "${envs[@]}"
do
    echo "Running environment $e"
    for ((i=2023;i<2031;i=i+parallel_seeds)); do
        for ((j=0;j<parallel_seeds && i+j<2031;j++)); do
            gpu_id=${available_gpus[$((j % num_gpus))]}
            seed=$((i+j))
            echo "Running seed $seed for $e on GPU $gpu_id in background"
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --Args.env $e --Args.buffer_type simple --Args.seed $seed --Args.use_tensorboard True >> "$log_dir/${e}_Seed_${seed}.log" 2>&1 &
        done
        wait
        echo "All seeds have completed for environment $e in range $i to $((i+parallel_seeds-1))"
    done
done

echo "All environments have completed"
