#!/bin/bash

GPU_IDS=(0 1)
MAX_PER_GPU=3

seed_begin=1
seed_end=5

ENV_NAME="HalfCheetah-v4"
TOTAL_STEPS=10000000
PROJECT_NAME="sb3-a2c-anpg-exp"

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

COMMON_HPARAMS=(
  "normalize:dict(norm_obs=True, norm_reward=True)"
  "n_envs:8"
  "n_steps:256"
  "gamma:0.99"
  "gae_lambda:1.0"
  "ent_coef:0.0"
  "vf_coef:0.5"
  "max_grad_norm:0.5"
  "log_param_norms:True"
  "separate_optimizers:True"
)

POLICY_BASE="dict(activation_fn=nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64])"

# Variant-specific hyperparameters (arrays are referenced via nameref in launch_variant).
A2C_PARAMS=(
  "learning_rate:1e-5"
  "actor_learning_rate:3e-4"
  "critic_learning_rate:3e-4"
  "normalize_advantage:True"
)

A2C_PULLBACK_PARAMS_LOGP=(
  "learning_rate:1e-5"
  "actor_learning_rate:5e-2"
  "critic_learning_rate:5e-4"
  "normalize_advantage:True"
  "use_pullback:True"
  "statistic:'logp'"
  "prox_h:0.1"
  "cg_lambda:0.1"
  "cg_max_iter:30"
  "cg_tol:1e-10"
  "fisher_ridge:0.1"
  "step_clip:0.05"
  "fr_order:1"
)

A2C_PULLBACK_PARAMS_SCORE=(
  "learning_rate:1e-5"
  "actor_learning_rate:5e-2"
  "critic_learning_rate:5e-4"
  "normalize_advantage:True"
  "use_pullback:True"
  "statistic:'score_per_dim'"
  "prox_h:0.1"
  "cg_lambda:0.1"
  "cg_max_iter:10"
  "cg_tol:1e-10"
  "fisher_ridge:0.1"
  "step_clip:0.05"
  "fr_order:1"
)

A2C_PULLBACK_PARAMS_LOGP2=(
  "learning_rate:1e-5"
  "actor_learning_rate:5e-2"
  "critic_learning_rate:5e-4"
  "normalize_advantage:True"
  "use_pullback:True"
  "statistic:'logp'"
  "prox_h:0.1"
  "cg_lambda:0.1"
  "cg_max_iter:30"
  "cg_tol:1e-10"
  "fisher_ridge:0.1"
  "step_clip:0.05"
  "fr_order:2"
)

source "$(dirname "$0")/anpg_variants_common.sh"

launch_variant() {
  local seed=$1
  local algo=$2
  local variant=$3
  local -n params_ref=$4  # nameref to pick the right array
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
        "${COMMON_HPARAMS[@]}" \
        "${params_ref[@]}"
}

for seed in $(seq ${seed_begin} ${seed_end}); do
  echo "===== seed ${seed} ====="
  launch_variant "${seed}" "a2c" "baseline" A2C_PARAMS
  launch_anpg_variants "${seed}"
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
