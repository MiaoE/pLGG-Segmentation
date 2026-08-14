#!/bin/bash
#SBATCH -N 1 -c 4
#SBATCH -G 1
#SBATCH --mem=16G --tmp=8G
#SBATCH --time=24:00:00

# submit as 
# sbatch run.sh

module load python/3.11.3_torch_gpu
source ./venv/bin/activate

python main_foundation_model.py -m SAM (or) MedSAM -f /path/to/dataset/folder
