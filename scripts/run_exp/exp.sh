#!/bin/bash

seed_begin=1
seed_end=1

for seed in `seq ${seed_begin} ${seed_end}`;
do
  echo "seed is ${seed}"
  set -x

       python train.py \
        --wandb-project-name sb3 \
        --wandb-run-extra-name $seed \
        --algo ppo \
        --env LunarLander-v3 \
        --vec-env subproc \
        --n-timesteps 5000000 \
        --seed 3 \
        --device auto \
        --track \
        --eval-freq 25000 \
        --eval-episodes 10 \
        --n-eval-envs 4 \
        --hyperparams \
            "n_envs:16" \
            "n_steps:1024" \
            "batch_size:64" \
            "n_epochs:4" \
            "learning_rate:3e-4" \
            "gamma:0.999" \
            "gae_lambda:0.98" \
            "clip_range:0.2" \
            "ent_coef:0.01" \
            "vf_coef:0.5" \
            "max_grad_norm:0.5" \
            "separate_optimizers:True" \
            "policy_kwargs:dict(activation_fn=nn.Tanh,
             net_arch=dict(pi=[64, 64], vf=[64, 64]), 
             optimizer_class=make_muon_with_aux_adam, 
             optimizer_kwargs=dict(muon_lr=0.02, adam_lr=3e-4, 
             muon_weight_decay=0.0, adam_weight_decay=0.0))"

done