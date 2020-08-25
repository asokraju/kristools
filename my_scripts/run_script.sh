#!/bin/bash -l
for test_name in 1
do
  for gamma in 0.6 0.9 0.99
  do
    for time_steps in 2 4 6 8
    do
      ./gpu_batch $gamma $time_steps
    done
  done
done
