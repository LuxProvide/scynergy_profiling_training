# scynergy_profiling_training

This repository provides scripts and examples to profile deep learning workloads on Meluxina GPU nodes using various profilers (Torch Profiler, cProfile, and py-spy).

## Prerequisites
- Access to the Meluxina cluster (login credentials and SSH key).
- `pyenv` and `pyenv-virtualenv` (or a compatible Python environment manager).
- `git` installed locally.

## 1. Clone the Repository
```bash
git clone git@github.com:LuxProvide/scynergy_profiling_training.git
cd scynergy_profiling_training
```

## 2. Reserve a GPU Node
Run the provided script to get an interactive GPU node for profiling:
```bash
source get_interactive_job_for_profiling.sh
```

## 3. Build the Python Virtual Environment
A helper script creates a virtual environment and installs all dependencies:
```bash
source buildVenv.sh
```

## 4. Activate the Virtual Environment
```bash
source activateTheVenv.sh
```

## 5. Verify Installation
Run a basic test to ensure CUDA and PyTorch see the GPU(s):
```bash
cd 0_BasicTest
source run_nvidia_smi.sh
```
You should see output similar to:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  |   00000000:03:00.0 Off |                    0 |
| N/A   39C    P0             57W /  400W |       1MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A100-SXM4-40GB          On  |   00000000:44:00.0 Off |                    0 |
| N/A   39C    P0             53W /  400W |       1MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA A100-SXM4-40GB          On  |   00000000:84:00.0 Off |                    0 |
| N/A   38C    P0             56W /  400W |       1MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA A100-SXM4-40GB          On  |   00000000:C4:00.0 Off |                    0 |
| N/A   39C    P0             57W /  400W |       1MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## 6. Example: MNIST Profiling
1. Change to the example directory:
   ```bash
   cd ../1_ExampleMnist/
   ```
2. Edit `launcher.sh` to choose:
   - Profiler type (Torch Profiler, cProfile, or no profiling).
   - Code version (`optimized` vs. `not_optimized`).
   - Number of dataloader workers and batch size.

   ```bash
   # In launcher.sh:
   # Here you can change the profiler, code version, number of workers, and batch size
   ```
3. Run the launcher:
   ```bash
   source launcher.sh
   ```

## 7. Accessing Profiling Results
For Torch Profiler or cProfile, the output is served via a local web server. On **your laptop**, open a new terminal and run:
```bash
ssh -p 8822 ${USER}@login.lxp.lu -i ~/.ssh/id_ed25519_mlux -NL 8080:$(hostname --ip-address):8080
```
Then open your browser at:
```
http://localhost:8080/
```

## 8. Using py-spy
`py-spy` generates a flamegraph file. You can:
- Download it to your local machine via `scp`:
  ```bash
  scp -P 8822 ${USER}@login.lxp.lu:/path/to/flamegraph.svg ./
  ```
- View it directly in VSCode or another IDE with remote file visualization.

## 9. Small contest: 

Now it’s time to apply what you have just learned. Move into the contest directory and explore the provided scripts:

```bash
cd ../2_Contest/
```

You will find two files:

`launcher.sh`: a Bash script to launch the contest workload

`script_with_hint.py`: a Python file containing hints for the task

The goal here is to:

* make a first run of the code

* profile the code as it is to get an idea of where time is being spent

* implement changes in the python code and the launcher to solve the bottlenecks, and profile again!

Do this until you are satisfied with the performance you obtained.

* Note that we put some hints on both the launcher and the python file.

* Also note that you can profile the code or not by adding the --profile flag to the python script. To ease comparison, you can switch between the optimized and not-optimized version of the code by adding the --optimized flag.

* We left other arguments that you can provide to the script to change its behaviour.

It's your turn!

## Contributing
Feel free to open issues or submit pull requests to improve scripts, add examples, or report bugs.


