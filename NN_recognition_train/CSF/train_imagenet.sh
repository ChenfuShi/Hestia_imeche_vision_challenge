#!/bin/bash --login
#$ -cwd
#$ -j y
#$ -o logs
#$ -l nvidia_v100
#$ -pe smp.pe 8
# qsub train.sh

# module load apps/anaconda3/5.2.0/bin # not needed

export OMP_NUM_THREADS=$NSLOTS

# check nvidia-smi
nvidia-smi 

rm /mnt/iusers01/jw01/mdefscs4/localscratch/imagenet_cache.tfdata -r
# rm /mnt/iusers01/jw01/mdefscs4/localscratch/* -r 
mkdir /mnt/iusers01/jw01/mdefscs4/localscratch/imagenet
cp /mnt/iusers01/jw01/mdefscs4/ra_challenge/imagenet/ILSVRC/Data/CLS-LOC/train ~/localscratch/imagenet/train -r -n


## activate conda environment
# this also includes cuda and cuda toolkits
source activate /mnt/jw01-aruk-home01/projects/psa_functional_genomics/differential_HiC_analyser/conda_env


# run python training script
cd /mnt/iusers01/jw01/mdefscs4/ra_challenge/hestia/Hestia_imeche_vision_challenge/NN_recognition_train
python imagenet_pretrainer.py 
