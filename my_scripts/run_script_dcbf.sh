#!/bin/bash -l
for test_name in dcbf_1
do
  for gamma in 0.99
  do
    for time_steps in 2
    do
      ./gpu_batch_dcbf.sh $test_name $gamma $time_steps
    done
  done
done
