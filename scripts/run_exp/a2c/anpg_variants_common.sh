#!/bin/bash

# Launch a fixed set of ANPG variants using existing base arrays
# A2C_PULLBACK_PARAMS_SCORE and A2C_PULLBACK_PARAMS_LOGP.
# Assumes launch_variant(seed, algo, variant_name, params_array_name) is defined in caller.
launch_anpg_variants() {
  local seed=$1
  local algo=${2:-a2c}

  # score_per_dim variants
  local score_kernel=("${A2C_PULLBACK_PARAMS_SCORE[@]}" "pb_use_nesterov_predict:True" "pb_use_kernel:True")
  local score_no_kernel=("${A2C_PULLBACK_PARAMS_SCORE[@]}" "pb_use_nesterov_predict:True" "pb_use_kernel:False" "pb_momentum_beta:0.3")

  # logp variants
  local logp_no_kernel=("${A2C_PULLBACK_PARAMS_LOGP[@]}" "pb_use_nesterov_predict:True" "pb_use_kernel:False" "pb_momentum_beta:0.3")
  local logp_kernel_beta08=("${A2C_PULLBACK_PARAMS_LOGP[@]}" "pb_use_nesterov_predict:True" "pb_use_kernel:True" "pb_momentum_beta:0.8")
  local logp_kernel_beta09=("${A2C_PULLBACK_PARAMS_LOGP[@]}" "pb_use_nesterov_predict:True" "pb_use_kernel:True" "pb_momentum_beta:0.9")
  local logp_kernel_beta02=("${A2C_PULLBACK_PARAMS_LOGP[@]}" "pb_use_nesterov_predict:True" "pb_use_kernel:True" "pb_momentum_beta:0.3")

  # launch_variant "${seed}" "${algo}" "pullback_score_kernel" score_kernel
  # launch_variant "${seed}" "${algo}" "pullback_score_nokernel" score_no_kernel
  # launch_variant "${seed}" "${algo}" "pullback_logp_nokernel" logp_no_kernel
  # launch_variant "${seed}" "${algo}" "pullback_logp_kernel_beta08" logp_kernel_beta08
  # launch_variant "${seed}" "${algo}" "pullback_logp_kernel_beta09" logp_kernel_beta09
  launch_variant "${seed}" "${algo}" "pullback_logp_kernel_beta02" logp_kernel_beta02
}
