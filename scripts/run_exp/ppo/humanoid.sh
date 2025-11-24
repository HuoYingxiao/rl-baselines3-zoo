#!/bin/bash

# 并行在多块 GPU 上跑多组 Humanoid-v4 的 PPO 实验
# 每块 GPU 同时不超过 MAX_PER_GPU 个任务

GPU_IDS=(0 1)            # 按机器实际情况修改
MAX_PER_GPU=2

seed_begin=1
seed_end=3

ENV_NAME="Humanoid-v4"
TOTAL_STEPS=30000000
PROJECT_NAME="sb3-ppo-optim"

# 需要 bash 支持关联数组
declare -A GPU_RUNNING   # 每块 GPU 当前正在运行的任务数
declare -A PID_TO_GPU    # 每个 pid 对应的 GPU
PIDS=()                  # 所有启动过的 pid 列表

# 初始化计数器
for gid in "${GPU_IDS[@]}"; do
  GPU_RUNNING[$gid]=0
done

# 更新当前所有还活着的任务信息：
# - 去掉已经结束的 pid
# - 重新统计各 GPU 上的任务数
refresh_counters() {
  local new_pids=()
  local pid gpu

  # 先清零
  for gid in "${GPU_IDS[@]}"; do
    GPU_RUNNING[$gid]=0
  done

  # 重新扫描所有 pids，看哪些还活着
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
      gpu=${PID_TO_GPU[$pid]}
      if [[ -n "$gpu" ]]; then
        GPU_RUNNING[$gpu]=$(( GPU_RUNNING[$gpu] + 1 ))
      fi
    else
      # 这个 pid 已经结束
      unset PID_TO_GPU[$pid]
    fi
  done

  PIDS=("${new_pids[@]}")
}

# 找一块还有空位的 GPU；如果当前都满了，就 sleep 一会再检查
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
    # 所有 GPU 都满载，等一会再看
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
  "normalize:True"
  "n_envs:1"
  "n_steps:1024"
  "batch_size:32"
  "gamma:0.95"
  "gae_lambda:0.9"
  "ent_coef:0.0002"
  "clip_range:0.3"
  "n_epochs:20"
  "max_grad_norm:2.0"
  "vf_coef:0.43"
  "log_param_norms:True"
  "separate_optimizers:True"
)

POLICY_BASE="dict(log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])"

POLICY_ADAM_NOWD="${POLICY_BASE}, optimizer_class=th.optim.Adam, optimizer_kwargs=dict(weight_decay=0.0))"
POLICY_ADAM_WD="${POLICY_BASE}, optimizer_class=th.optim.Adam, optimizer_kwargs=dict(weight_decay=0.01))"
POLICY_MUON_NOWD="${POLICY_BASE}, optimizer_class=make_muon_with_aux_adam, optimizer_kwargs=dict(muon_lr=5e-4, adam_lr=5e-5, muon_weight_decay=0.0, adam_weight_decay=0.0, muon_momentum=0.95))"
POLICY_MUON_WD="${POLICY_BASE}, optimizer_class=make_muon_with_aux_adam, optimizer_kwargs=dict(muon_lr=5e-4, adam_lr=5e-5, muon_weight_decay=0.02, adam_weight_decay=0.01, muon_momentum=0.95))"
POLICY_MUON_NOWD_NOMOM="${POLICY_BASE}, optimizer_class=make_muon_with_aux_adam, optimizer_kwargs=dict(muon_lr=5e-4, adam_lr=5e-5, muon_weight_decay=0.0, adam_weight_decay=0.0, muon_momentum=0.0))"
POLICY_SGD="${POLICY_BASE}, optimizer_class=th.optim.SGD, optimizer_kwargs=dict(momentum=0.9, weight_decay=0.0))"

launch_variant() {
  local seed=$1
  local run_suffix=$2
  local policy_kwargs=$3
  local lr_value=$4
  local extra_name="ppo_${run_suffix}_seed${seed}"

  run_with_gpu \
    python train.py \
      --wandb-project-name "${PROJECT_NAME}" \
      --wandb-run-extra-name "${extra_name}" \
      --algo ppo \
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
        "learning_rate:${lr_value}" \
        "policy_kwargs:${policy_kwargs}" \
        "log_param_norms:True"
}

for seed in $(seq ${seed_begin} ${seed_end}); do
  echo "===== seed ${seed} ====="
  launch_variant "${seed}" "adam_nowd"        "${POLICY_ADAM_NOWD}"        "5e-5"
  launch_variant "${seed}" "adam_wd"         "${POLICY_ADAM_WD}"          "5e-5"
  launch_variant "${seed}" "muon_nowd"       "${POLICY_MUON_NOWD}"        "5e-4"
  launch_variant "${seed}" "muon_wd"         "${POLICY_MUON_WD}"          "5e-4"
  launch_variant "${seed}" "muon_nowd_nomom" "${POLICY_MUON_NOWD_NOMOM}"  "5e-5"
  launch_variant "${seed}" "sgd"             "${POLICY_SGD}"              "3e-3"
done

# 等所有还活着的子进程结束
if ((${#PIDS[@]} > 0)); then
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
    fi
  done
fi

echo "All jobs finished."
