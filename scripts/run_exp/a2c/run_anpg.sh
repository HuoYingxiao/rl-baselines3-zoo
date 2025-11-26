#!/bin/bash

seed_begin=2
seed_end=2

ENV_NAME="Hopper-v4"
TOTAL_STEPS=10000000
PROJECT_NAME="sb3-anpg-t"

for seed in $(seq ${seed_begin} ${seed_end}); do
  echo "===== seed ${seed} : A2C + ANPG (dense G pullback) ====="
  set -x

  # A2C + ANPG pullback（dense G）
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
    --log-interval 1 \
    --hyperparams \
      "n_envs:8" \
      "n_steps:256" \
      "learning_rate:5e-5" \
      "actor_learning_rate:1e-2" \
      "critic_learning_rate:3e-4" \
      "gamma:0.99" \
      "gae_lambda:1.0" \
      "ent_coef:0.0" \
      "vf_coef:0.5" \
      "max_grad_norm:0.5" \
      "normalize_advantage:True" \
      "use_rms_prop:False" \
      "use_pullback:True" \
      "statistic:'score_per_dim'" \
      "prox_h:1.0" \
      "fr_order:1" \
      "cg_lambda:0.1" \
      "cg_max_iter:25" \
      "cg_tol:1e-10" \
      "fisher_ridge:0.1" \
      "step_clip:0.01" \
      "policy_kwargs:dict(activation_fn=nn.Tanh, net_arch=[64, 64])" \
      "log_param_norms:True" \
      "separate_optimizers:True" \
      "pb_use_inner_loop:False" \
      "pb_inner_steps:5" \
      "pb_inner_lr:0.00005" \
      "pb_use_kernel:True" \
      "pb_kernel_num_anchors:32" \
      "pb_kernel_sigma:1.0" \

  set +x
  #   echo "===== seed ${seed} : vanilla A2C ====="
  # set -x
  # python train.py \
  #   --wandb-project-name "${PROJECT_NAME}" \
  #   --wandb-run-extra-name "a2c_seed${seed}" \
  #   --algo a2c \
  #   --env "${ENV_NAME}" \
  #   --vec-env subproc \
  #   --n-timesteps ${TOTAL_STEPS} \
  #   --seed ${seed} \
  #   --device auto \
  #   --track \
  #   --eval-freq 25000 \
  #   --eval-episodes 10 \
  #   --n-eval-envs 4 \
  #   --log-interval 1 \
  #   --hyperparams \
  #     "n_envs:8" \
  #     "n_steps:256" \
  #     "learning_rate:1e-4" \
  #     "actor_learning_rate:3e-4" \
  #     "critic_learning_rate:3e-4" \
  #     "gamma:0.99" \
  #     "gae_lambda:1.0" \
  #     "ent_coef:0.0" \
  #     "vf_coef:0.5" \
  #     "max_grad_norm:0.5" \
  #     "normalize_advantage:True" \
  #     "use_rms_prop:False" \
  #     "policy_kwargs:dict(activation_fn=nn.Tanh, net_arch=[64, 64])" \
  #     "log_param_norms:True" \
  #     "separate_optimizers:True"

  # set +x
done
