#!/bin/bash

GPU_IDS=(0 1)
MAX_PER_GPU=3   # 每块 GPU 最多同时跑多少个进程

seed_begin=1
seed_end=3

ENV_NAME="Hopper-v4"
TOTAL_STEPS=10000000
PROJECT_NAME="sb3-anpg"

# 要搜索的超参数
PROX_HS=(0.01 0.1 1.0 10.0 50.0)
ANCHOR_LIST=(4 32 64 128)

declare -A GPU_RUNNING
declare -A PID_TO_GPU
PIDS=()

for gid in "${GPU_IDS[@]}"; do
  GPU_RUNNING[$gid]=0
done

refresh_counters() {
  local new_pids=()
  local pid gpu

  # 先清零计数
  for gid in "${GPU_IDS[@]}"; do
    GPU_RUNNING[$gid]=0
  done

  # 遍历当前 PIDS，保留还活着的，同时更新每块 GPU 上的计数
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

# 通用超参数
COMMON_HPARAMS=(
  "normalize:dict(norm_obs=True, norm_reward=True)"
  "n_envs:8"
  "n_steps:256"
  "gamma:0.99"
  "gae_lambda:0.95"
  "ent_coef:0.001"
  "vf_coef:0.5"
  "max_grad_norm:0.5"
  "log_param_norms:True"
  "separate_optimizers:True"
)

# policy + Adam
POLICY_BASE="dict(log_std_init=-2, ortho_init=False, activation_fn=nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64])"
POLICY_ADAM="${POLICY_BASE}, optimizer_class=th.optim.Adam, optimizer_kwargs=dict(weight_decay=0.0))"

launch_anpg_variant() {
  local seed=$1
  local prox_h=$2
  local anchors=$3

  # 把 prox_h 里的点换成 p，方便在 wandb 里看
  local prox_tag=${prox_h//./p}
  local extra_name="a2c_anpg_ph${prox_tag}_na${anchors}_seed${seed}"

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
      --log-interval 1 \
      --hyperparams \
        "${COMMON_HPARAMS[@]}" \
        "learning_rate:1e-5" \
        "actor_learning_rate:1e-1" \
        "critic_learning_rate:1e-4" \
        "policy_kwargs:${POLICY_ADAM}" \
        "normalize_advantage:True" \
        "use_pullback:True" \
        "statistic:'score_per_dim'" \
        "prox_h:${prox_h}" \
        "fr_order:1" \
        "cg_lambda:0.01" \
        "cg_max_iter:10" \
        "cg_tol:1e-10" \
        "fisher_ridge:0.1" \
        "step_clip:0.01" \
        "pb_use_inner_loop:False" \
        "pb_inner_steps:5" \
        "pb_inner_lr:0.00005" \
        "pb_use_kernel:True" \
        "pb_kernel_num_anchors:${anchors}" \
        "pb_kernel_sigma:1.0"
}

# 主循环：扫 prox_h × anchors × seeds
for prox_h in "${PROX_HS[@]}"; do
  for anchors in "${ANCHOR_LIST[@]}"; do
    for seed in $(seq ${seed_begin} ${seed_end}); do
      echo "===== prox_h=${prox_h}, anchors=${anchors}, seed=${seed} ====="
      launch_anpg_variant "${seed}" "${prox_h}" "${anchors}"
    done
  done
done

# 等待所有子进程结束
if ((${#PIDS[@]} > 0)); then
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
    fi
  done
fi

echo "All jobs finished."
