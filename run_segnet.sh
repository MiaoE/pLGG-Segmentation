#!/bin/bash
#SBATCH -N 1 -c 8
#SBATCH --gres=gpu:Tesla_V100-PCIE-32GB:1
#SBATCH --mem=64G --tmp=32G
#SBATCH --time=7-0
module load python/3.11.3_torch_gpu
source ./venv/bin/activate

python ./segnet.py