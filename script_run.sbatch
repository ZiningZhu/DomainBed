#!/bin/bash

#SBATCH -c 8
#SBATCH -p t4v2,t4v1,rtx6000,p100
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --mem=16G
#SBATCH --output=slurm/%j.out

module load transformers4
#python ~/env_scripts/python_test.py

# Script to download data
#python -m domainbed.scripts.download --data_dir=./domainbed/data

python3 -m domainbed.scripts.train --data_dir=./domainbed/data/MNIST/ --algorithm GroupDRO --dataset RotatedMNIST --test_env 2 --output_dir GroupDRO_RotatedMNIST_output

python3 -m domainbed.scripts.probe --checkpoint_dir=GroupDRO_RotatedMNIST_output/ --report_freq 500 --slurm_id $SLURM_JOB_ID