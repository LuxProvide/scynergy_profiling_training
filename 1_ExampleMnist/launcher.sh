#!/bin/bash -l
cd ..
source ActivateTheVenv.sh
cd - 

####################################################################################
#Here you can change the profiler you want to test or choose to run the code without profiling 
#You can also pick the "optimized" and not optimized version of the code 
#Alos, we left two important parameters you can play with which are the number of workers that will be involved in the dataloader as well as the batch sizeo
#Experiment the impact of these parameters on the CPU and GPU utilization

# choose between PROFILER=torch   or PROFILER=py-spy   or PROFILER=cprofile or Profile r=NoProfiler 
PROFILER="NoProfiler"
OPTIMIZED=true
# Parameters you can play with
BATCHSIZE=128
NWORKERS=4
####################################################################################

run_options="--epochs=3 "
script="quick_inefficient_profiled.py"

PROFILER=${PROFILER:-torch}   # default to "torch" if unset
OPTIMIZED=${OPTIMIZED:-false} # default to false

# OPTION: add "--optimized" if requested
if [ "$OPTIMIZED" = "true" ]; then
    run_options="${run_options} --optimized --nworkers=${NWORKERS} --batchsize=${BATCHSIZE}"
    srun_options="-n1 -c ${NWORKERS}"
else
    srun_options="-n1 -c1"
fi

if [ "$PROFILER" = "torch" ]; then
    # OPTION 1: torch profiler
    run_options="${run_options} --torchprofile"
    EXEC="python ${script} ${run_options}"

elif [ "$PROFILER" = "py-spy" ]; then
    # OPTION 2: py-spy
    outputDir="../firstExample/py-spy"
    mkdir -p "${outputDir}"
    EXEC="py-spy record -o ${outputDir}/profiling.svg -- python ${script} ${run_options}"

elif [ "$PROFILER" = "cprofile" ]; then
    # OPTION 3: built-in cProfile flag
    run_options="${run_options} --useCProfile"
    EXEC="python ${script} ${run_options}"

elif [ "$PROFILER" = "NoProfiler" ]; then
    run_options="${run_options}"
    EXEC="python ${script} ${run_options}"

else
    echo "Unknown PROFILER: ${PROFILER}"
    echo "Valid values are: torch, py-spy, cprofile, NoProfiler"
    # exit 1
fi

echo "Running with profiler = ${PROFILER}"
echo "Command → ${EXEC}"
eval "srun ${srun_options} ${EXEC}"
