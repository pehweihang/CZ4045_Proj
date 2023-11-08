#!/bin/bash

##Job Script

#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --job-name=NLP
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
source activate nlp-proj
python src/train.py -m \
  epochs=500 \
  batch_size=1,4,8,16,32,64 \
  lr=0.001,0.0001,0.00001 \
  model.n_layers=1,2,5,10 \
  model.n_hidden=1,5,10,25,50,100 \
  model.dropout=0,0.1,0.15,0.2
