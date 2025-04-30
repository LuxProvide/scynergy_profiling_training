import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import os, multiprocessing
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity
from torch.amp import autocast, GradScaler

def training(args):
    # Heavy CPU transforms: RandomCrop + RandomHorizontalFlip  
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),              # CPU-bound augmentation :contentReference[oaicite:0]{index=0}
        transforms.RandomCrop(32, padding=4),           # CPU-bound augmentation :contentReference[oaicite:1]{index=1}
        transforms.ToTensor(),                          # PIL→Tensor :contentReference[oaicite:2]{index=2}
        transforms.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5)),          # per-channel normalization :contentReference[oaicite:3]{index=3}
    ])

    """
    if args.optimized:
        # Hint: You can try to allow for tf32 matrix multiplication to try to leverage the GPU tensor core 
    """

    """
    Hint: where is the data stored (on which data tier)? 
    Do I have access to a faster tier?
    """
    if args.optimized:
        # Change this and if you have access to a faster data tier, move the data there
        rootDataDir=None
    else:
        rootDataDir='/project/home/p200865/2_Contest/data'

    # 2) Real dataset: CIFAR-10 (50k train, 10k test; 3×32×32 images) 
    train_ds = datasets.CIFAR10(root=rootDataDir, train=True,
                                download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root=rootDataDir, train=False,
                                download=True, transform=transform)


    print(f"The training set has {len(train_ds)} images and the test one {len(test_ds)}")

    allowed = len(os.sched_getaffinity(0))
    print(f"I detect {allowed} cores")

    if not args.optimized:
        # single-threaded, no pinning
        nworkers = 0 
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=64, shuffle=True,
            num_workers=0, pin_memory=False
        )
        test_loader  = torch.utils.data.DataLoader(
            test_ds,  batch_size=64, shuffle=False,
            num_workers=0, pin_memory=False
        )
    else:
        '''
        Hint:
        - Will the batch size have an influence on my perf?
        - What about the number of workers? Can I avoid synchronous data loading 

        !!! you can control the number of available cores with srun -c 
        
        
        - what about the prefetch factor ?
        '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=False)        # or pretrained=True if you want to use the default weight 
    # Replace final fc so it outputs 10 classes instead of 1000
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    opt    = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit   = nn.CrossEntropyLoss().to(device)

    if args.torchprofile:
        """
        we wrote this to make your life easier when it comes to retrieving profiling results 
        """
        from datetime import datetime
        timestamp=datetime.today().strftime('%Y-%m-%d')
        output_profiler_name=f"resnet50_CIFAR10_fine_tuning_{nworkers}_workers"
        if args.optimized:
            output_profiler_name += "_optimized"
        output_profiler_name += "_" + timestamp
        username=os.getenv('USER')
        output_dir_trace=Path(f"output/{output_profiler_name}").resolve()
        output_dir_trace.mkdir(parents=True, exist_ok=True)

        
    nwait = 1
    nwarmup=1
    nactive=5
    nrepeat=3

    
    from contextlib import nullcontext
    """
    profiling will take place only if --profile is given as a flag when running the script 
    """
    prof_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir_trace),
            schedule=torch.profiler.schedule(wait=nwait, warmup=nwarmup, active=nactive, repeat=nrepeat),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,# do not set to True 
            with_modules=True,) if args.torchprofile else nullcontext()

    t0_global = time.time()

    with prof_ctx as prof:
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            running_loss = 0.0
            for i, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss    = crit(outputs, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print(f"[Epoch {epoch+1}, Batch {i}]  loss: {loss.item():.4f}")

                if args.torchprofile:
                    prof.step()

            print(f"Epoch {epoch+1} done in {time.time()-t0:.2f}s; avg loss {running_loss/len(train_loader):.4f}")


            # Simple test pass
            model.eval()
            correct = 0
            total   = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    preds = model(imgs).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total   += labels.size(0)
            print(f"Test Accuracy: {100*correct/total:.2f}%\n")


    print(f"Total running time: {time.time()-t0_global:.2f}s")


    if args.torchprofile:
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

        print(f"Run:\n\n tensorboard --logdir={output_dir_trace} --host 0.0.0.0 --port 8080 --load_fast=false")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int,   default=3)
    parser.add_argument('--batch',  type=int,   default=128)
    # Suggested flags you can provide to your script
    parser.add_argument('--nworkers',  type=int,   default=4)
    parser.add_argument('--batchsize',  type=int,   default=64)
    parser.add_argument('--torchprofile', action='store_true')
    parser.add_argument('--optimized', action='store_true')
    args = parser.parse_args()

    training(args)
