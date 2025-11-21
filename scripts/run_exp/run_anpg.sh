#!/bin/bash

seed_begin=1
seed_end=3

ENV_NAME="LunarLander-v3"
TOTAL_STEPS=5000000
PROJECT_NAME="sb3-a2c-anpg"

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
      "policy_kwargs:dict(activation_fn=nn.Tanh, net_arch=[64, 64])" \
      "log_param_norms:True"

  set +x
    echo "===== seed ${seed} : vanilla A2C ====="
  set -x
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
      "policy_kwargs:dict(activation_fn=nn.Tanh, net_arch=[64, 64])" \
      "log_param_norms:True"

  set +x
done
