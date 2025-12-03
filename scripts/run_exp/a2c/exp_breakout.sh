#!/bin/bash
set -euo pipefail

# Breakout RAM ANPG/A2C experiments with simple multi-GPU scheduling.
GPU_IDS=(0 1)
MAX_PER_GPU=2

seed_begin=1
seed_end=2

ENV_NAME="ALE/Breakout-v5"
TOTAL_STEPS=5000000
PROJECT_NAME="sb3-anpg-t"

declare -A GPU_RUNNING
declare -A PID_TO_GPU
PIDS=()

for gid in "${GPU_IDS[@]}"; do
  GPU_RUNNING[$gid]=0
done

trap 'echo "Stopping..."; kill 0' SIGINT SIGTERM

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
    sleep 5
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

COMMON_HPARAMS=(
  "n_envs:16"
  "n_steps:256"
  "gamma:0.99"
  "gae_lambda:0.95"
  "ent_coef:0.01"
  "vf_coef:0.5"
  "max_grad_norm:0.5"
  "normalize_advantage:True"
  "frame_stack:1"
  "log_param_norms:True"
  "separate_optimizers:True"
  "policy:'MlpPolicy'"
  "policy_kwargs:dict(activation_fn=nn.ReLU, net_arch=[256, 256])"
)

A2C_BASELINE=(
  "learning_rate:7e-4"
  "actor_learning_rate:7e-4"
  "critic_learning_rate:7e-4"
)

A2C_ANPG_LOGP=(
  "learning_rate:7e-4"
  "actor_learning_rate:1e-2"
  "critic_learning_rate:3.5e-4"
  "use_pullback:True"
  "normalize_advantage:True"
  "n_critic_updates:20"
  "statistic:'logp'"
  "prox_h:1.0"
  "fr_order:1"
  "cg_lambda:0.1"
  "cg_max_iter:10"
  "cg_tol:1e-10"
  "fisher_ridge:0.01"
  "step_clip:0.05"
  "pb_use_kernel:True"
  "pb_kernel_num_anchors:4"
  "pb_kernel_sigma:1.0"
  "pb_use_nesterov_predict:True"
  "pb_predict_iters:1"
  "pb_use_momentum:False"
  "pb_momentum_beta:0.3"
)

PPO_BASELINE=(
  "n_envs:16"
  "n_steps:256"
  "batch_size:1024"
  "gamma:0.99"
  "gae_lambda:0.95"
  "ent_coef:0.01"
  "clip_range:0.1"
  "n_epochs:4"
  "vf_coef:0.5"
  "max_grad_norm:0.5"
  "normalize:dict(norm_obs=True, norm_reward=True)"
  "log_param_norms:True"
  "separate_optimizers:True"
  "policy:'MlpPolicy'"
  "policy_kwargs:dict(activation_fn=nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256]))"
  "learning_rate:2.5e-4"
  "actor_learning_rate:2.5e-4"
  "critic_learning_rate:2.5e-4"
)

run_variant() {
  local seed=$1
  local extra_name=$2
  local algo=$3
  shift 3
  run_with_gpu \
    python train.py \
      --wandb-project-name "${PROJECT_NAME}" \
      --wandb-run-extra-name "${extra_name}" \
      --algo "${algo}" \
      --env "${ENV_NAME}" \
      --vec-env subproc \
      --env-kwargs "obs_type:'ram'" \
      --n-timesteps ${TOTAL_STEPS} \
      --seed ${seed} \
      --device auto \
      --track \
      --eval-freq 25000 \
      --eval-episodes 10 \
      --n-eval-envs 4 \
      --log-interval 1 \
      --hyperparams \
        "${COMMON_HPARAMS[@]}" \
        "$@"
}

for seed in $(seq ${seed_begin} ${seed_end}); do
#   echo "===== seed ${seed} : Breakout RAM A2C baseline ====="
#   run_variant "${seed}" "breakoutram_a2c_seed${seed}" a2c "${A2C_BASELINE[@]}"

#   echo "===== seed ${seed} : Breakout RAM A2C + ANPG (logp, kernel) ====="
#   run_variant "${seed}" "breakoutram_anpg_logp_seed${seed}" a2c "${A2C_ANPG_LOGP[@]}"

  echo "===== seed ${seed} : Breakout RAM PPO baseline ====="
  run_variant "${seed}" "breakoutram_ppo_seed${seed}" ppo "${PPO_BASELINE[@]}"
done

if ((${#PIDS[@]} > 0)); then
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
    fi
  done
fi
