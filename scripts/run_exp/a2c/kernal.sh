#!/bin/bash

# -----------------------------
# GPU 配置
# -----------------------------
GPU_IDS=(0 1)
MAX_PER_GPU=3

seed_begin=1
seed_end=3

ENV_NAME="CartPole-v1"
TOTAL_STEPS=1000000   
PROJECT_NAME="sb3-a2c-anpg-pendulum-kernels"

declare -A GPU_RUNNING
declare -A PID_TO_GPU
PIDS=()

for gid in "${GPU_IDS[@]}"; do
  GPU_RUNNING[$gid]=0
done

refresh_counters() {
  local new_pids=()
  local pid gpu

  # 重新清零计数
  for gid in "${GPU_IDS[@]}"; do
    GPU_RUNNING[$gid]=0
  done

  # 统计仍在运行的进程
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

# -----------------------------
# 公共超参数
# -----------------------------
COMMON_HPARAMS=(
  "normalize:dict(norm_obs=True, norm_reward=True)"
  "n_envs:4"        
  "n_steps:256"
  "gamma:0.99"
  "gae_lambda:1.0"
  "ent_coef:0.0"
  "vf_coef:0.5"
  "max_grad_norm:0.5"
  "log_param_norms:True"
  "separate_optimizers:True"
)

# policy 结构
# 在 rl-zoo 的 hyperparams 里一般写成：
# policy_kwargs:\"dict(activation_fn=nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))\"
POLICY_KWARG="policy_kwargs:dict(activation_fn=nn.Tanh, net_arch=dict(pi=[64,64], vf=[64,64]))"

# -----------------------------
# 各种算法 / kernel 变体
# -----------------------------

# 基础 A2C（无 pullback）
A2C_BASE=(
  "learning_rate:7e-4"
  "actor_learning_rate:5e-4"
  "critic_learning_rate:0.0005"
  "normalize_advantage:True"
  "${POLICY_KWARG}"
)

# ANPG + FR 特征（无 kernel）
A2C_PULLBACK_FR=(
  "learning_rate:1e-4"
  "actor_learning_rate:5e-2"
  "critic_learning_rate:0.00025"
  "normalize_advantage:True"
  "use_pullback:True"
  "n_critic_updates:20"
  "statistic:'score_per_dim'"
  "prox_h:0.2"
  "cg_lambda:0.1"
  "cg_max_iter:20"
  "cg_tol:1e-10"
  "fisher_ridge:0.1"
  "step_clip:0.01"
  "pb_use_kernel:False"
  "pb_kernel_num_anchors:4"
  "pb_kernel_sigma:1.0"
  "${POLICY_KWARG}"
)

A2C_PULLBACK_LOGP_FR=(
  "learning_rate:1e-4"
  "actor_learning_rate:5e-2"
  "critic_learning_rate:0.00025"
  "normalize_advantage:True"
  "use_pullback:True"
  "n_critic_updates:20"
  "statistic:'logp'"
  "prox_h:0.2"
  "cg_lambda:0.1"
  "cg_max_iter:20"
  "cg_tol:1e-10"
  "fisher_ridge:0.1"
  "step_clip:0.01"
  "pb_use_kernel:False"
  "pb_kernel_num_anchors:4"
  "pb_kernel_sigma:1.0"
  "${POLICY_KWARG}"
)

# ANPG + kernel（RBF）
A2C_PULLBACK_K_RBF=(
  "learning_rate:1e-4"
  "actor_learning_rate:5e-2"
  "critic_learning_rate:0.00025"
  "normalize_advantage:True"
  "use_pullback:True"
  "n_critic_updates:20"
  "statistic:'score_per_dim'"
  "prox_h:0.2"
  "cg_lambda:0.1"
  "cg_max_iter:20"
  "cg_tol:1e-10"
  "fisher_ridge:0.1"
  "step_clip:0.01"
  "pb_use_kernel:True"
  "pb_kernel_type:'rbf'"
  "pb_kernel_num_anchors:4"
  "pb_kernel_sigma:1.0"
  "${POLICY_KWARG}"
)

A2C_PULLBACK_LOGP_K_RBF=(
  "learning_rate:1e-4"
  "actor_learning_rate:5e-2"
  "critic_learning_rate:0.00025"
  "normalize_advantage:True"
  "use_pullback:True"
  "n_critic_updates:20"
  "statistic:'logp'"
  "prox_h:0.2"
  "cg_lambda:0.1"
  "cg_max_iter:20"
  "cg_tol:1e-10"
  "fisher_ridge:0.1"
  "step_clip:0.01"
  "pb_use_kernel:True"
  "pb_kernel_type:'rbf'"
  "pb_kernel_num_anchors:4"
  "pb_kernel_sigma:1.0"
  "${POLICY_KWARG}"
)

# ANPG + kernel（Laplace）
A2C_PULLBACK_K_LAPLACE=(
  "learning_rate:1e-4"
  "actor_learning_rate:5e-2"
  "critic_learning_rate:0.00025"
  "normalize_advantage:True"
  "use_pullback:True"
  "n_critic_updates:20"
  "statistic:'score_per_dim'"
  "prox_h:0.2"
  "cg_lambda:0.1"
  "cg_max_iter:20"
  "cg_tol:1e-10"
  "fisher_ridge:0.1"
  "step_clip:0.01"
  "pb_use_kernel:True"
  "pb_kernel_type:'laplace'"
  "pb_kernel_num_anchors:4"
  "pb_kernel_sigma:1.0"
  "${POLICY_KWARG}"
)

# ANPG + kernel（多项式 kernel）
A2C_PULLBACK_K_POLY=(
  "learning_rate:1e-4"
  "actor_learning_rate:5e-2"
  "critic_learning_rate:0.00025"
  "normalize_advantage:True"
  "use_pullback:True"
  "n_critic_updates:20"
  "statistic:'score_per_dim'"
  "prox_h:0.2"
  "cg_lambda:0.1"
  "cg_max_iter:20"
  "cg_tol:1e-10"
  "fisher_ridge:0.1"
  "step_clip:0.01"
  "pb_use_kernel:True"
  "pb_kernel_type:'poly'"
  "pb_kernel_num_anchors:4"
  "pb_kernel_sigma:1.0"   # 如果代码里 poly 不用 sigma 也没有问题
  "${POLICY_KWARG}"
)

# -----------------------------
# 启动一个变体
# -----------------------------
launch_variant() {
  local seed=$1
  local variant=$2
  local -n params_ref=$3

  local extra_name="pendulum_${variant}_seed${seed}"

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
      --eval-freq 5000 \
      --eval-episodes 10 \
      --n-eval-envs 1 \
      --log-interval 1 \
      --hyperparams \
        "${COMMON_HPARAMS[@]}" \
        "${params_ref[@]}"
}

# -----------------------------
# 主循环：多 seed，多 kernel
# -----------------------------
for seed in $(seq ${seed_begin} ${seed_end}); do
  echo "===== seed ${seed} ====="
  launch_variant "${seed}" "a2c-base"       A2C_BASE
  launch_variant "${seed}" "pb-fr"          A2C_PULLBACK_FR
  launch_variant "${seed}" "pb--logp-fr"    A2C_PULLBACK_LOGP_FR
  launch_variant "${seed}" "pb-k-rbf"       A2C_PULLBACK_K_RBF
  launch_variant "${seed}" "pb-k-logp-rbf"  A2C_PULLBACK_LOGP_K_RBF
  launch_variant "${seed}" "pb-k-laplace"   A2C_PULLBACK_K_LAPLACE
  launch_variant "${seed}" "pb-k-poly"      A2C_PULLBACK_K_POLY
done

# -----------------------------
# 等待所有任务结束
# -----------------------------
if ((${#PIDS[@]} > 0)); then
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
    fi
  done
fi

echo "All jobs finished."
