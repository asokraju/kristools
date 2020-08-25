#!/bin/bash

TEST_NAME=$1
GAMMA=$2
TS=$3


PARENT_DIR="$(dirname $PWD)"             #file is inside this  directory
EXEC_DIR=$PWD                            #gpu_batch script is inside this dir
TEST_NAME_DIR="test_name=${TEST_NAME}"   #directory with test name
GAMMA_DIR="gamma=${GAMMA}"               #directory for parameter gamma name
TS_DIR="time_steps=${TS}"                #directory for parameter time steps name

mkdir -p $TEST_NAME_DIR                  #making a directory with test name
RESULTS_DIR=${EXEC_DIR}/${TEST_NAME_DIR} #Directory for results
cd $RESULTS_DIR                          #we are inside the results_dir

mkdir -p $GAMMA_DIR                      #making a directory for parameter gamma name
cd $GAMMA_DIR
mkdir -p $TS_DIR                         #making a directory for parameter time steps name
cd $TS_DIR

export run_exec=$PARENT_DIR/microgrid_ddpg.py
#export run_exec=/afs/crc.nd.edu/user/k/kkosaraj/kristools/microgrid_dcbf.py
export run_flags="--gamma=${GAMMA} --time_steps=${TS} --summary_dir='$PWD/' > out.txt"  

echo "#!/bin/bash" > job.sh
echo "#$ -q gpu" >> job.sh
echo "#$ -l gpu_card=1" >>job.sh
#echo "#$ -pe smp 1" >> job.sh
echo "#$ -N DDPG_gamma=${GAMMA}_TS=${TS}" >> job.sh
echo "#$ -o info" >> job.sh
echo "module load conda" >> job.sh
echo "module load cuda" >> job.sh
echo "module load cudnn" >> job.sh
echo "conda activate tf_gpu_krishna" >> job.sh

echo "python $run_exec $run_flags" >> job.sh

qsub job.sh


#DO YOU SEE ME