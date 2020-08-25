#!/bin/bash -l
for test_name in dcbf_1
do
  for gamma in 0.5 0.9 0.99
  do
    for time_steps in 2 4 6
    do
      ./gpu_batch_dcbf.sh $test_name $gamma $time_steps
    done
  done
done
