#!/bin/bash

# Simple TRPO sweep on Hopper with a few seeds.

seed_begin=1
seed_end=1

ENV_NAME="Hopper-v4"
TOTAL_STEPS=10000000
PROJECT_NAME="sb3-anpg-t"

POLICY_KWARGS="dict(activation_fn=nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))"

COMMON_HPARAMS=(
  "normalize:dict(norm_obs=True, norm_reward=True)"
  "n_envs:8"
  "n_steps:256"
  "batch_size:2048"
  "gamma:0.99"
  "gae_lambda:0.95"
  "normalize_advantage:True"
  "target_kl:0.01"
  "cg_max_steps:10"
  "cg_damping:0.1"
  "line_search_shrinking_factor:0.8"
  "line_search_max_iter:10"
  "n_critic_updates:20"
  "learning_rate:5e-4"
  "policy_kwargs:${POLICY_KWARGS}"
)

for seed in $(seq ${seed_begin} ${seed_end}); do
  echo "===== TRPO Hopper seed ${seed} ====="
  python train.py \
    --wandb-project-name "${PROJECT_NAME}" \
    --wandb-run-extra-name "trpo_hopper_seed${seed}" \
    --algo trpo \
    --env "${ENV_NAME}" \
    --vec-env subproc \
    --n-timesteps ${TOTAL_STEPS} \
    --seed ${seed} \
    --device auto \
    --track \
    --eval-freq 25000 \
    --eval-episodes 10 \
    --n-eval-envs 4 \
    --hyperparams "${COMMON_HPARAMS[@]}"
done

echo "TRPO Hopper sweep finished."
