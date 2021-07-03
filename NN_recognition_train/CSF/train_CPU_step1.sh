#!/bin/bash --login
#$ -cwd
#$ -j y
#$ -o logs
##$ -l nvidia_v100
#$ -pe smp.pe 16
# qsub -v model_name=initial_mobilenet train.sh

# module load apps/anaconda3/5.2.0/bin # not needed

export OMP_NUM_THREADS=$NSLOTS

# check nvidia-smi
# nvidia-smi 
lscpu 

## activate conda environment
# this also includes cuda and cuda toolkits
source activate /mnt/jw01-aruk-home01/projects/psa_functional_genomics/differential_HiC_analyser/conda_env


# run python training script
cd /mnt/iusers01/jw01/mdefscs4/ra_challenge/hestia/Hestia_imeche_vision_challenge/NN_recognition_train
python train_CPU_step1.py ${model_name}
