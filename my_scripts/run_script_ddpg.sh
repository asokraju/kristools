#!/bin/bash -l
for test_name in ddpg_corrected_v2
do
  for gamma in 0.99
  do
    for time_steps in 2
    do
      for random_seed in 200 300 400 500 600 700
      do
        ./gpu_batch_ddpg.sh $test_name $gamma $time_steps $random_seed
      done
    done
  done
done