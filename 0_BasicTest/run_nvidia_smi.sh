#!/bin/bash 
cd ..
source activateTheVenv.sh
cd -
srun nvidia-smi
