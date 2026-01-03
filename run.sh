#!/bin/bash

# submit as 
# sbatch run.sh

module load python/3.12
source ./venv/bin/activate

python main_foundation_model.py -f /path/to/dataset/folder
