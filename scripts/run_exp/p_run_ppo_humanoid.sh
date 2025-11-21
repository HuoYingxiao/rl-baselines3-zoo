#!/bin/bash

# 并行在多块 GPU 上跑多组 Humanoid-v4 的 PPO 实验
# 每块 GPU 同时不超过 3 个任务

GPU_IDS=(0 1)            # 按机器实际情况修改
MAX_PER_GPU=3

seed_begin=1
seed_end=3

ENV_NAME="Humanoid-v4"
TOTAL_STEPS=10000000
PROJECT_NAME="sb3-ppo-optim"

# 需要 bash 4+ 支持关联数组
declare -A GPU_RUNNING
declare -A PID_TO_GPU
running=0

for gid in "${GPU_IDS[@]}"; do
  GPU_RUNNING[$gid]=0
done

wait_for_one_job() {
  local finished_pid finished_gpu
  finished_pid=$(wait -n)
  finished_gpu=${PID_TO_GPU[$finished_pid]}
  if [[ -n "$finished_gpu" ]]; then
    GPU_RUNNING[$finished_gpu]=$(( GPU_RUNNING[$finished_gpu] - 1 ))
    unset PID_TO_GPU[$finished_pid]
  fi
  running=$(( running - 1 ))
}

acquire_gpu() {
  local gpu_id
  while true; do
    for gpu_id in "${GPU_IDS[@]}"; do
      if (( GPU_RUNNING[$gpu_id] < MAX_PER_GPU )); then
        echo "$gpu_id"
        return 0
      fi
    done
    wait_for_one_job
  done
}

run_with_gpu() {
  local gpu_id
  gpu_id=$(acquire_gpu)
  echo "Launching on GPU ${gpu_id}: $*"

  CUDA_VISIBLE_DEVICES=${gpu_id} "$@" &
  local pid=$!
  PID_TO_GPU[$pid]=$gpu_id
  GPU_RUNNING[$gpu_id]=$(( GPU_RUNNING[$gpu_id] + 1 ))
  running=$(( running + 1 ))
}

# 通用超参（Humanoid-v4 官方建议的 PPO 配置为基础）
COMMON_HPARAMS=(
  "normalize:True"
  "n_envs:1"
  "n_steps:1024"
  "batch_size:256"
  "gamma:0.95"
  "gae_lambda:0.9"
  "ent_coef:0.0024"
  "clip_range:0.3"
  "n_epochs:5"
  "max_grad_norm:2.0"
  "vf_coef:0.43"
  "log_param_norms:True"
)

POLICY_BASE="dict(log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])"

# 各优化器对应的 policy_kwargs
POLICY_ADAM_NOWD="${POLICY_BASE}, optimizer_class=th.optim.Adam, optimizer_kwargs=dict(lr=3e-4, weight_decay=0.0))"
POLICY_ADAM_WD="${POLICY_BASE}, optimizer_class=th.optim.Adam, optimizer_kwargs=dict(lr=3e-4, weight_decay=0.01))"
POLICY_MUON_NOWD="${POLICY_BASE}, optimizer_class=make_muon_with_aux_adam, optimizer_kwargs=dict(muon_lr=0.02, adam_lr=3e-4, muon_weight_decay=0.0, adam_weight_decay=0.0, muon_momentum=0.95))"
POLICY_MUON_WD="${POLICY_BASE}, optimizer_class=make_muon_with_aux_adam, optimizer_kwargs=dict(muon_lr=0.02, adam_lr=3e-4, muon_weight_decay=0.02, adam_weight_decay=0.01, muon_momentum=0.95))"
POLICY_MUON_NOWD_NOMOM="${POLICY_BASE}, optimizer_class=make_muon_with_aux_adam, optimizer_kwargs=dict(muon_lr=0.02, adam_lr=3e-4, muon_weight_decay=0.0, adam_weight_decay=0.0, muon_momentum=0.0))"
POLICY_SGD="${POLICY_BASE}, optimizer_class=th.optim.SGD, optimizer_kwargs=dict(lr=3e-3, momentum=0.9, weight_decay=0.0))"

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
  launch_variant "${seed}" "adam_nowd" "${POLICY_ADAM_NOWD}" "3e-4"
  launch_variant "${seed}" "adam_wd" "${POLICY_ADAM_WD}" "3e-4"
  launch_variant "${seed}" "muon_nowd" "${POLICY_MUON_NOWD}" "2e-2"
  launch_variant "${seed}" "muon_wd" "${POLICY_MUON_WD}" "2e-2"
  launch_variant "${seed}" "muon_nowd_nomom" "${POLICY_MUON_NOWD_NOMOM}" "2e-2"
  launch_variant "${seed}" "sgd" "${POLICY_SGD}" "3e-3"
done

while (( running > 0 )); do
  wait_for_one_job
done

echo "All jobs finished."
