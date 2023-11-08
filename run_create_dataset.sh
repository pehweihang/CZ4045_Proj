#!/bin/bash

##Job Script

#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --job-name=NLP
#SBATCH --output=/home/FYP/pehw0013/CZ4045_Proj/slurm_out/output_%x_%j.out
#SBATCH --error=/home/FYP/pehw0013/CZ4045_Proj/slurm_out/error_%x_%j.err

module load anaconda
source activate nlp-proj
python src/create_dataset.py
