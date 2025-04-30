#!/bin/bash
module load env/release/2023.1
module load Python/3.11
baseDir=$PWD
python3 -m venv EnvTrainingScynergy311
# by doing this, I ensure that the venv does not use user site-packages
echo 'export PYTHONNOUSERSITE=1' >>EnvTrainingScynergyPy311/bin/activate
unset PYTHONPATH
unset PYTHONUSERBASE
unset PIP_PREFIX
source EnvTrainingScynergyPy311/bin/activate
python -m site
pip install --isolated --no-cache-dir -r requirements.txt
