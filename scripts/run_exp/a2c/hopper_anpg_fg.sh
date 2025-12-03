#!/bin/bash

# Foreground launcher: same experiments as hopper_anpg.sh, but keeps the script
# in the foreground while running multiple jobs across GPUs concurrently.

# List of GPUs to use and per-GPU concurrency.
GPU_IDS=(0 1)
MAX_PER_GPU=1

seed_begin=1
seed_end=1

ENV_NAME="Hopper-v4"
TOTAL_STEPS=5000000
PROJECT_NAME="sb3-anpg"

declare -A GPU_RUNNING     # gpu_id -> running count
declare -A PID_TO_GPU      # pid -> gpu_id
PIDS=()                    # tracked pids

for gid in "${GPU_IDS[@]}"; do
  GPU_RUNNING[$gid]=0
done

refresh_counters() {
  local new_pids=()
  local pid gpu

  # reset counts
  for gid in "${GPU_IDS[@]}"; do
    GPU_RUNNING[$gid]=0
  done

  # keep only live pids
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

# Find a GPU with spare slots; wait if all are full.
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

  echo ">>> Launching on GPU ${gpu_id}: $*"
  CUDA_VISIBLE_DEVICES=${gpu_id} "$@" &
  pid=$!

  PID_TO_GPU[$pid]=$gpu_id
  PIDS+=("$pid")
  GPU_RUNNING[$gpu_id]=$(( GPU_RUNNING[$gpu_id] + 1 ))
}

# Common hyperparameters shared by both vanilla A2C and pullback A2C.
COMMON_HPARAMS=(
  "normalize:dict(norm_obs=True, norm_reward=True)"
  "n_envs:4"
  "n_steps:512"
  "gamma:0.99"
  "gae_lambda:1.0"
  "ent_coef:0.0"
  "vf_coef:0.5"
  "max_grad_norm:0.5"
  "use_rms_prop:True"
  "rms_prop_eps:1e-5"
  "log_param_norms:True"
)

# Recommended policy backbone + optimizer choices.
POLICY_BASE="dict(log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])"
POLICY_ADAM="${POLICY_BASE}, optimizer_class=th.optim.Adam, optimizer_kwargs=dict(weight_decay=0.0))"

# Variant-specific hyperparameters.
A2C_PARAMS=(
  "learning_rate:1e-3"
  "actor_learning_rate:1e-3"
  "critic_learning_rate:0.001"
  "policy_kwargs:${POLICY_ADAM}"
  "normalize_advantage:True"
)

A2C_PULLBACK_PARAMS=(
  "learning_rate:5e-3"
  "actor_learning_rate:5e-3"
  "critic_learning_rate:0.0025"
  "policy_kwargs:${POLICY_ADAM}"
  "normalize_advantage:True"
  "use_pullback:True"
  "n_critic_updates:20"
  "statistic:'logp'"
  "prox_h:5.0"
  "cg_lambda:0.01"
  "cg_max_iter:10"
  "cg_tol:1e-10"
  "fisher_ridge:0.1"
  "step_clip:0.05"
)

run_variant() {
  local seed=$1
  local variant=$2
  local -n params_ref=$3  # nameref to pick the right array
  local extra_name="a2c_${variant}_seed${seed}"

  run_with_gpu \
    python train.py \
      --wandb-project-name "${PROJECT_NAME}" \
      --wandb-run-extra-name "${extra_name}" \
      --algo a2c \
      --env "${ENV_NAME}" \
      --vec-env subproc \
      --n-timesteps ${TOTAL_STEPS} \
      --seed ${seed} \
      --device auto \
      --track \
      --eval-freq 25000 \
      --eval-episodes 10 \
      --n-eval-envs 4 \
      --hyperparams \
        "${COMMON_HPARAMS[@]}" \
        "${params_ref[@]}"
}

for seed in $(seq ${seed_begin} ${seed_end}); do
  echo "===== seed ${seed} ====="
  run_variant "${seed}" "pullback" A2C_PULLBACK_PARAMS
  run_variant "${seed}" "baseline" A2C_PARAMS
done

# Wait for outstanding jobs (keeps script in foreground).
if ((${#PIDS[@]} > 0)); then
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
    fi
  done
fi

echo "All jobs finished."
