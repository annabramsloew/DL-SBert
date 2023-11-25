#!/bin/sh 


### General LSF options 
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J train_siamese_epochs1

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00

# request 30GB of system-memory
#BSUB -R "rusage[mem=30GB]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u anryg@dtu.dk

### -- send notification at start --
#BSUB -B

### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi

### -- end of LSF options --

#ENV_REPO = /zhome/49/1/147319/sbert
#REPO = /zhome/49/1/147319/sbert/DL-SBert/Scripts


# Load environment variables
#source ./.env


# Create job_out if it is not present
if [[ ! -d /zhome/49/1/147319/sbert/DL-SBert/Scripts/job_out ]]; then
	mkdir /zhome/49/1/147319/sbert/DL-SBert/Scripts/job_out
fi


date=$(date +%Y%m%d_%H%M)
mkdir /zhome/49/1/147319/sbert/DL-SBert/Scripts/runs/train/${date}


# Activate venv
module load python3/3.10.12
module load cuda/12.1
source /zhome/49/1/147319/sbert/.venv/bin/activate


# Exit if previous command failed
if [[ $? -ne 0 ]]; then
	exit 1
fi


# run training
python3 /zhome/49/1/147319/sbert/DL-SBert/Scripts/train_triplet.py --epochs=2 --train_batch_size=64