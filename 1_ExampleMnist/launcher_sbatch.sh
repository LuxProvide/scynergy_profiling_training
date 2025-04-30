#!/bin/bash -l
#SBATCH --job-name=profilingA
#SBATCH --ntasks=1
#SBATCH -A p200865
#SBATCH -p gpu
#SBATCH -q default
#SBATCH --time=00:30:00
#SBATCH --output=../output/%u/%x_%j.out   # %x is the job name, %j is the job ID
#SBATCH --error=../output/%u/%x_%j.err    # Same for stderr
#SBATCH --nodes=1
#SBATCH --disable-perfparanoid
#SBATCH --nodelist=mel[2012,2016-2020,2022,2025-2027,2031,2036,2047,2050-2052,2054,2056,2058,2059,2062,2063,2068,2073,2082-2084,2090,2092,2095,2106,2108,2114-2116,2123-2126,2128,2132,2140,2147,2154,2155,2157,2158,2160,2161,2179,2181-2183,2185,2187,2191,2192,2198-2200]

source launcher.sh
