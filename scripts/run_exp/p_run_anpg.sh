#!/bin/bash
#chmod +x scripts/exp_a2c_anpg_parallel.sh
# bash scripts/exp_a2c_anpg_parallel.sh > exp_a2c_anpg.log 2>&1 &

# 可用的 GPU 列表
GPU_IDS=(0 1)            # 按你的机器改，比如只有一块就写 (0)
MAX_PER_GPU=3            # 每块 GPU 同时最多跑 3 个任务

seed_begin=1
seed_end=3

ENV_NAME="LunarLander-v3"
TOTAL_STEPS=5000000
PROJECT_NAME="sb3-a2c-anpg"

# 需要 bash 4+ 支持关联数组
declare -A GPU_RUNNING   # 每块 GPU 当前在跑的任务数
declare -A PID_TO_GPU    # 记录每个 pid 对应哪块 GPU
running=0                # 总共在跑的任务数

# 初始化计数
for gid in "${GPU_IDS[@]}"; do
  GPU_RUNNING[$gid]=0
done

# 等待一个任务结束，并更新对应 GPU 的计数
wait_for_one_job() {
  local finished_pid finished_gpu
  finished_pid=$(wait -n)          # 等任意一个后台任务结束
  finished_gpu=${PID_TO_GPU[$finished_pid]}
  if [[ -n "$finished_gpu" ]]; then
    GPU_RUNNING[$finished_gpu]=$(( GPU_RUNNING[$finished_gpu] - 1 ))
    unset PID_TO_GPU[$finished_pid]
  fi
  running=$(( running - 1 ))
}

# 获取一块有空位的 GPU，没有空位就阻塞等任务结束
acquire_gpu() {
  local gpu_id
  while true; do
    for gpu_id in "${GPU_IDS[@]}"; do
      if (( GPU_RUNNING[$gpu_id] < MAX_PER_GPU )); then
        echo "$gpu_id"
        return 0
      fi
    done
    # 没有任何 GPU 有空位，就先等一个任务结束
    wait_for_one_job
  done
}

# 启动一个任务：传入 "命令..." 这一整串
run_with_gpu() {
  local gpu_id cmd
  gpu_id=$(acquire_gpu)
  echo "Launching on GPU ${gpu_id}: $*"

  CUDA_VISIBLE_DEVICES=${gpu_id} "$@" &
  local pid=$!

  PID_TO_GPU[$pid]=$gpu_id
  GPU_RUNNING[$gpu_id]=$(( GPU_RUNNING[$gpu_id] + 1 ))
  running=$(( running + 1 ))
}

for seed in $(seq ${seed_begin} ${seed_end}); do
  echo "===== seed ${seed} : A2C + ANPG (dense G pullback) ====="

  # A2C + ANPG
  run_with_gpu \
    python train.py \
      --wandb-project-name "${PROJECT_NAME}" \
      --wandb-run-extra-name "a2c_anpg_seed${seed}" \
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
        "n_envs:16" \
        "n_steps:5" \
        "learning_rate:7e-4" \
        "gamma:0.99" \
        "gae_lambda:1.0" \
        "ent_coef:0.0" \
        "vf_coef:0.5" \
        "max_grad_norm:0.5" \
        "normalize_advantage:True" \
        "use_rms_prop:False" \
        "use_pullback:True" \
        "statistic:'logp'" \
        "prox_h:10.0" \
        "cg_lambda:0.01" \
        "cg_max_iter:10" \
        "cg_tol:1e-10" \
        "fisher_ridge:0.1" \
        "step_clip:0.1" \
        "policy_kwargs:dict(activation_fn=nn.Tanh, net_arch=[64, 64])"

  echo "===== seed ${seed} : vanilla A2C ====="

  # 普通 A2C
  run_with_gpu \
    python train.py \
      --wandb-project-name "${PROJECT_NAME}" \
      --wandb-run-extra-name "a2c_seed${seed}" \
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
        "n_envs:16" \
        "n_steps:5" \
        "learning_rate:7e-4" \
        "gamma:0.99" \
        "gae_lambda:1.0" \
        "ent_coef:0.0" \
        "vf_coef:0.5" \
        "max_grad_norm:0.5" \
        "normalize_advantage:True" \
        "use_rms_prop:False" \
        "policy_kwargs:dict(activation_fn=nn.Tanh, net_arch=[64, 64])"
done

# 等所有后台任务结束
while (( running > 0 )); do
  wait_for_one_job
done

echo "All jobs finished."
