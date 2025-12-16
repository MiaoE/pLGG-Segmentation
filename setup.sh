#!/bin/bash
set -e  # exit on error

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing base requirements..."
pip install -r requirements.txt

echo "Installing Segment Anything..."
pip install git+https://github.com/facebookresearch/segment-anything.git

echo "Cloning MedSAM..."
if [ ! -d "MedSAM" ]; then
    git clone https://github.com/bowang-lab/MedSAM
fi

echo "Installing MedSAM..."
cd MedSAM
pip install -e .
cd ..

echo "Setup complete"