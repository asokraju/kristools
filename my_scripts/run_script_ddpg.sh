#!/bin/bash -l
for test_name in ddpg_1
do
  for gamma in 0.9 0.99
  do
    for time_steps in 2 4 6
    do
      ./gpu_batch_ddpg.sh $test_name $gamma $time_steps
    done
  done
done
