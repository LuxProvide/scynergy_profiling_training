#!/bin/bash -l
source ../EnvTrainingScynergyPy311/bin/activate

run_options="--epochs=3 --batchsize=128 --nworkers=4 "
script="quick_inefficient_profiled.py"

PROFILER="NoProfiler"
OPTIMIZED=true

# e.g. export PROFILER=torch   or PROFILER=py-spy   or PROFILER=cprofile
PROFILER=${PROFILER:-torch}   # default to "torch" if unset
OPTIMIZED=${OPTIMIZED:-false} # default to false

# OPTION: add "--optimized" if requested
if [ "$OPTIMIZED" = "true" ]; then
    run_options="${run_options} --optimized --nworkers=64"
    srun_options="-n1 -c64"
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
