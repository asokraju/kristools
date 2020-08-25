#!/bin/bash

TEST_NAME=$1
GAMMA=$2
TS=$3

#Creating necessary folders to save the results of the experiment
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

export run_exec=$PARENT_DIR/microgrid_dcbf.py #python script that we want to run
#export run_exec=/afs/crc.nd.edu/user/k/kkosaraj/kristools/microgrid_dcbf.py
export run_flags="--gamma=${GAMMA} --time_steps=${TS} --summary_dir='$PWD' > out.txt"   #flags for the script

echo "#!/bin/bash" > job.sh                         
echo "#$ -M kkosaraj@nd.edu" >> job.sh  # Email address for job notification
echo "#$ -m abe"   >> job.sh         # Send mail when job begins, ends and aborts
echo "#$ -q gpu" >> job.sh                           # which queue to use: debug, long, gpu
echo "#$ -l gpu_card=1" >>job.sh                       # need if we use gpu queue
#echo "#$ -pe smp 1" >> job.sh
echo "#$ -N DCBF_gamma=${GAMMA}_TS=${TS}" >> job.sh   # name for job
echo "#$ -o info" >> job.sh
echo "module load conda" >> job.sh                   #loading the desired modules
echo "module load cuda" >> job.sh
echo "module load cudnn" >> job.sh
echo "conda activate tf_gpu_krishna" >> job.sh

echo "python $run_exec $run_flags" >> job.sh

qsub job.sh


#DO YOU SEE ME