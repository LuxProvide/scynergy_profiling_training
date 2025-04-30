#!/bin/bash -l

cd .. 
source activateTheVenv.sh 
cd - 

# you might have to change this one too. Run one taks with -n 1 but control the number of cores assignes with -c
srun python script_with_hint.py --torchprofile
