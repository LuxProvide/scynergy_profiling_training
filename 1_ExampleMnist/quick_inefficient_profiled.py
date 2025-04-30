import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.profiler import profile, record_function, ProfilerActivity
import os, multiprocessing
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def show_ssh_command():
    import subprocess
    bash_snippet = r'''
    echo -e "\n\n\n"
    echo "-------------------------------------------------------------------------------------------"
    echo "                         IMPORTANT: run this on your LAPTOP LOCAL TERMINAL                                                "
    echo "ssh -p 8822 ${USER}@login.lxp.lu  -i ~/.ssh/id_ed25519_mlux -NL 8080:$(hostname --ip-address):8080"
    echo "-------------------------------------------------------------------------------------------"
    echo "                                         IMPORTANT                                                "
    echo -e "\n\n\n"
    '''

    # Run under bash so that -e, ${USER}, and $(...) get interpreted correctly
    subprocess.run(
        bash_snippet,
        shell=True,
        executable="/bin/bash"
    )

def training(args):
    # 1. Define a heavy data augmentation pipeline (slow transformations on CPU)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),    # expensive resize & crop
        transforms.RandomHorizontalFlip(),                     # augmentation (fast)
        transforms.ColorJitter(brightness=0.5, contrast=0.5, 
                            saturation=0.5, hue=0.1),       # color adjustments (CPU heavy)
        transforms.GaussianBlur(kernel_size=5, sigma=2.0),      # simulate costly filter
        transforms.ToTensor()                                   # convert PIL image to tensor
    ])

    # 2. Create a fake dataset of 1000 images (3x224x224) with the heavy transform
    dataset = datasets.FakeData(size=2000, image_size=(3, 224, 224), 
                                num_classes=10, transform=transform)

    allowed = len(os.sched_getaffinity(0))
    print(f"I detect {allowed} cores")
    if args.optimized:
        allowed = len(os.sched_getaffinity(0))
        # figure out how many cores you're actually allowed
        nworkers = min(args.nworkers, allowed)    
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=True, 
                                            num_workers=nworkers, pin_memory=True)
    else:
        nworkers = 1
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=True, 
                                            num_workers=0, pin_memory=False)

    print(f"There are {nworkers} workers")

    # 3. Define a simple model (small CNN) and training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        nn.Linear(32, 10)
    ).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    if args.torchprofile:
        from datetime import datetime
        timestamp=datetime.today().strftime('%Y-%m-%d')
        output_profiler_name=f"FirstExample_{nworkers}_workers"
        if args.optimized:
            output_profiler_name += "_optimized"
        output_profiler_name += "_" + timestamp
        username=os.getenv('USER')
        output_dir_trace=Path(f"{output_profiler_name}").resolve()
        output_dir_trace.mkdir(parents=True, exist_ok=True)


    nwait = 1
    nwarmup=1
    nactive=5
    nrepeat=1

    # turn your training loop into an explicit iterator
    loader_it = iter(loader)
    num_batches = len(loader)

    
    from contextlib import nullcontext
    # pick your profiler ctx only if requested
    prof_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir_trace),
            schedule=torch.profiler.schedule(wait=nwait, warmup=nwarmup, active=nactive, repeat=nrepeat),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,# do not set to True 
            with_modules=True,) if args.torchprofile else nullcontext()

    running_loss = 0

        
    with prof_ctx as prof:    

        model.train()
        epoch_start = time.time()

        i = 0
        loader_it = iter(loader)
            
        if args.torchprofile:
            # create a writer; by default it will write into ./runs/<timestamp>/
            # writer = SummaryWriter(log_dir=output_dir_trace)
            from torch.profiler import record_function
            for _ in range(num_batches):
                with record_function("DataLoader.Next"):
                    images, labels = next(loader_it)

                with record_function("ToDevice"):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                with record_function("Forward"):
                    outputs = model(images)

                with record_function("Loss"):
                    loss = criterion(outputs, labels)

                with record_function("Backward+Step"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                i += 1 
                # with record_function("Logging"):
                #     writer.add_scalar("loss", loss.item(), i )

                running_loss += loss.item()

                if i % 100 == 0:
                    print(f"[Batch {i}]  loss: {loss.item():.4f}")

                prof.step()
        else:
            for _ in range(num_batches):
                images, labels = next(loader_it)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1 
                if i % 10 == 0:
                    print(f"[Batch {i}]  loss: {loss.item():.4f}")


        if args.torchprofile:
            print(prof.key_averages(group_by_stack_n=5)
                .table(sort_by="self_cpu_time_total", row_limit=10))                

                

            show_ssh_command()

            print(f"Run:\n\n tensorboard --logdir={output_dir_trace} --host 0.0.0.0 --port 8080 --load_fast=false")


        epoch_time = time.time() - epoch_start
        print(f"Epoch completed in {epoch_time:.2f} seconds")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int,   default=3)
    parser.add_argument('--nworkers',  type=int,   default=4)
    parser.add_argument('--batchsize',  type=int,   default=64)
    parser.add_argument('--torchprofile', action='store_true')
    parser.add_argument('--optimized', action='store_true')
    parser.add_argument('--useCProfile', action='store_true')
    args = parser.parse_args()

    if args.useCProfile:
        import cProfile, io, pstats
        pr = cProfile.Profile()
        # Only profile the block between enable() and disable()
        pr.enable()
        training(args)                  
        pr.disable()
        
        # dump stats to stdout
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumtime')
        ps.print_stats(15)         # top 20 cumulative entries
        print(s.getvalue())

        output_profiler_name=f"cProfile"
        username=os.getenv('USER')
        
        from pathlib import Path

        output_dir_trace = Path(f"FirstExample/{output_profiler_name}")
        output_dir_trace.mkdir(parents=True, exist_ok=True)
        dump_path = output_dir_trace / 'my_profile.prof'
        pr.dump_stats(str(dump_path))
        print(f"Profile written to {dump_path}")

        show_ssh_command()
        print("Profiling is now over. Run the following command on Meluxina:")
        print(f"python -m snakeviz --server --hostname 0.0.0.0 --port 8080 {dump_path}") 

    else:
        training(args)
