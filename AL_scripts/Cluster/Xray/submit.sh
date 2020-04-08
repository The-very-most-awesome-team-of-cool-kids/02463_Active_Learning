#!/bin/sh
### General options
###  specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J ActiveLearning
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 20:00
# request xGB of system-memory
#BSUB -R "rusage[mem=20GB]"
### -- set the email address --
##BSUB -u s183920@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo %J_output.out
#BSUB -eo %J_error.err
# -- end of LSF options --

module load python3/3.6.7
### to find torch version used: "torch.version.cuda"
module load cuda/9.0

cd ../..
source active_learning-env/bin/activate

### to find version used "pkg.__version__"
python -m pip install torch==1.0.0 torchvision==0.2.1 matplotlib==3.0.3 tensorflow==1.13.1 keras==2.2.4 kaggle

### Remember to set correct save paths in the scripts
python AL_scripts/run_AL.py