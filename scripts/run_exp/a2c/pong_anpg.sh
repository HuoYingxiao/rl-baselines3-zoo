#!/bin/bash

GPU_IDS=(0 1)
MAX_PER_GPU=3

seed_begin=1
seed_end=5

ENV_NAME="PongNoFrameskip-v4"
TOTAL_STEPS=10000000
PROJECT_NAME="sb3-a2c-anpg"

declare -A GPU_RUNNING
declare -A PID_TO_GPU
PIDS=()

for gid in "${GPU_IDS[@]}"; do
  GPU_RUNNING[$gid]=0
done

refresh_counters() {
  local new_pids=()
  local pid gpu

  for gid in "${GPU_IDS[@]}"; do
    GPU_RUNNING[$gid]=0
  done

  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
      gpu=${PID_TO_GPU[$pid]}
      if [[ -n "$gpu" ]]; then
        GPU_RUNNING[$gpu]=$(( GPU_RUNNING[$gpu] + 1 ))
      fi
    else
      unset PID_TO_GPU[$pid]
    fi
  done

  PIDS=("${new_pids[@]}")
}

acquire_gpu() {
  local gpu_id
  while true; do
    refresh_counters
    for gpu_id in "${GPU_IDS[@]}"; do
      if (( GPU_RUNNING[$gpu_id] < MAX_PER_GPU )); then
        echo "$gpu_id"
        return 0
      fi
    done
    sleep 10
  done
}

run_with_gpu() {
  local gpu_id pid
  gpu_id=$(acquire_gpu)

  echo "Launching on GPU ${gpu_id}: $*"
  CUDA_VISIBLE_DEVICES=${gpu_id} "$@" &
  pid=$!

  PID_TO_GPU[$pid]=$gpu_id
  PIDS+=("$pid")
  GPU_RUNNING[$gpu_id]=$(( GPU_RUNNING[$gpu_id] + 1 ))
}

# 这部分是 A2C 和 PPO 都通用的环境超参数
COMMON_ENV_PARAMS=(
  "normalize:dict(norm_obs=False, norm_reward=True)"
  "n_envs:4"
  "n_steps:1024"
  "gamma:0.99"
  "gae_lambda:0.98"
  "ent_coef:0.01"
  "vf_coef:0.5"
  "max_grad_norm:0.5"
  "policy_kwargs:dict(activation_fn=nn.Tanh, net_arch=[64, 64])" 
)

# ===== A2C 相关超参数 =====

A2C_PARAMS=(
  "learning_rate:1e-5"
  "actor_learning_rate:5e-4"
  "critic_learning_rate:5e-4"
  "normalize_advantage:True"
  "log_param_norms:True"
  "separate_optimizers:True"
)


A2C_PULLBACK_PARAMS1=(
  "learning_rate:1e-5"
  "actor_learning_rate:1e-1"
  "critic_learning_rate:5e-4"
  "normalize_advantage:True"
  "log_param_norms:True"
  "separate_optimizers:True"
  "use_pullback:True"
  "statistic:'score_per_dim'"
  "prox_h:1.0"
  "cg_lambda:0.1"
  "cg_max_iter:30"
  "cg_tol:1e-10"
  "fisher_ridge:0.1"
  "step_clip:0.01"
  "pb_use_kernel:True" \
  "pb_kernel_num_anchors:32" \
  "pb_kernel_sigma:1.0" \
)

# ===== PPO baseline 超参数 =====
# 这里给一个比较常规的配置，可以再按需要调
PPO_PARAMS=(
  "learning_rate:3e-4"
  "batch_size:64"
  "n_epochs:10"
  "gamma:0.99"
  "gae_lambda:0.98"
  "ent_coef:0.01"
  "vf_coef:0.5"
  "clip_range:0.2"
)

launch_variant() {
  local seed=$1
  local algo=$2
  local variant=$3
  local -n params_ref=$4  # nameref
  local extra_name="${algo}_${variant}_seed${seed}"

  run_with_gpu \
    python train.py \
      --wandb-project-name "${PROJECT_NAME}" \
      --wandb-run-extra-name "${extra_name}" \
      --algo "${algo}" \
      --env "${ENV_NAME}" \
      --vec-env subproc \
      --n-timesteps ${TOTAL_STEPS} \
      --seed ${seed} \
      --device auto \
      --track \
      --eval-freq 25000 \
      --eval-episodes 10 \
      --n-eval-envs 4 \
      --log-interval 1 \
      --hyperparams \
        "${COMMON_ENV_PARAMS[@]}" \
        "${params_ref[@]}"
}

for seed in $(seq ${seed_begin} ${seed_end}); do
  echo "===== seed ${seed} ====="

  # A2C baselines + pullback
  launch_variant "${seed}" "a2c" "baseline"        A2C_PARAMS
  launch_variant "${seed}" "a2c" "pullback_score"  A2C_PULLBACK_PARAMS1

  # PPO baseline
  launch_variant "${seed}" "ppo" "baseline"        PPO_PARAMS
done

# Wait for outstanding jobs.
if ((${#PIDS[@]} > 0)); then
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
    fi
  done
fi

echo "All jobs finished."
