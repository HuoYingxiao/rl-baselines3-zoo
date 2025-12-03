#!/bin/bash

# Quick ANPG/A2C experiments on Breakout RAM
seed_begin=1
seed_end=2

ENV_NAME="Breakout-ramNoFrameskip-v4"
TOTAL_STEPS=5000000
PROJECT_NAME="sb3-a2c-anpg-breakout-ram-debug"

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
  "fisher_ridge:0.1"
  "step_clip:0.05"
  "pb_use_kernel:True"
  "pb_kernel_num_anchors:4"
  "pb_kernel_sigma:1.0"
  "pb_use_nesterov_predict:True"
  "pb_predict_iters:1"
  "pb_use_momentum:False"
  "pb_momentum_beta:0.3"
)

run_variant() {
  local seed=$1
  local extra_name=$2
  shift 2
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
      "$@"
}

for seed in $(seq ${seed_begin} ${seed_end}); do
  echo "===== seed ${seed} : Breakout RAM A2C baseline ====="
  run_variant "${seed}" "breakoutram_a2c_seed${seed}" "${A2C_BASELINE[@]}"

  echo "===== seed ${seed} : Breakout RAM A2C + ANPG (logp, kernel) ====="
  run_variant "${seed}" "breakoutram_anpg_logp_seed${seed}" "${A2C_ANPG_LOGP[@]}"
done

