import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import time

from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import spawn
import torch.distributed as dist

from models.GPT2 import GPT2LMHeadModel
from models.Lora import LoRA_GPT2
from utils.data_utils import get_config, get_weight, get_loader, get_subset, pred_text
from utils.model_utils import run_forward, run_backward, model_split_transformer
from utils.scheduler import WarmupCosineSchedule, WarmupConstantSchedule, WarmupLinearSchedule


import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def setup_gpt_models(args):
    # initialize GPT2 model
    model = GPT2LMHeadModel(get_config('./datasets/gpt2_config.json'))
    model = get_weight(model, './datasets/gpt2_pretrained.pth')

    # create a Combined LoRA ViT Model for demo
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    # draw GPT2 model structure with  dummy testcase
    # input_ids = torch.randint(0, 50257, (1024,))
    # position_ids = torch.range(0, 1023, dtype=torch.long)
    # dummy_testcase = (input_ids, position_ids, input_ids)
    # writer.add_graph(model, dummy_testcase)

    # convert GPT model to LoRA-GPT
    LoRA_GPT2(gpt_model=model, r=4, alpha=8)

    # split LoRA GPT Model
    return model_split_transformer(base_model=model, split_block_id=args.num_split)


def train(rank, args, model, dataloader):
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    if rank==0:
        train_server(args, model['model_sv'], dataloader, writer)
    elif rank==1:
        train_client(args, model['model_cl'], dataloader, writer)

    dist.destroy_process_group()

########################## Local Device ##########################
def train_client(args, model, dataloader, writer:SummaryWriter):
    # process exclusively occupy device
    torch.cuda.set_device(args.device_cl)

    # initialize model, data, recv_tensor, context manager and gloabl_step, 
    model.to(args.device_cl)
    trainloader, testloader = dataloader
    backward_grad = torch.empty([args.batch_size,args.max_length,768]).to(args.device_cl)
    inputs_ctx, outputs_ctx = [], []
    global_step = 0

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, total_steps=args.num_steps)
    elif args.decay_type == "linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, total_steps=args.num_steps)
    else:
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    optimizer.zero_grad()

    trainloader = iter(trainloader)
    model.train()
    for step in range(len(trainloader)+1):
        if step!=len(trainloader):
            inputs, positions = next(trainloader)
            inputs = inputs.to(args.device_cl)
            positions = positions.to(args.device_cl)

            # Forward Client
            forward_feature, _ = run_forward(model, input_ids=inputs, position_ids=positions)
            inputs_ctx.append(inputs)
            outputs_ctx.append(forward_feature)

            # Exchange feature and gradient
            dist.broadcast(forward_feature,1)

        if step>0:
            # Exchange feature and gradient
            dist.broadcast(backward_grad,0)

            # Backward Client
            inputs = inputs_ctx.pop(0)
            forward_feature = outputs_ctx.pop(0)
            run_backward(outputs = forward_feature,
                        grad_tensor = backward_grad,
                        input_tensor = inputs,
                        )
            
            # Update Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Validation
        if global_step%args.num_interval == 0:
            valid_client(args, model, testloader, writer)

        global_step += 1
        # Update tensorboard
        writer.add_scalar("train/lr_cl", scalar_value=optimizer.param_groups[0]['lr'], global_step=global_step)
        
        if global_step % args.num_steps == 0:
            break

    writer.close()


def valid_client(args, model, testloader, writer:SummaryWriter):
    model.eval()
    
    testloader = get_subset(testloader, args.num_test)

    for batch in testloader:
        inputs, positions = batch
        inputs = inputs.to(args.device_cl)
        positions = positions.to(args.device_cl)

        # forward_client
        forward_feature, _ = model(inputs, position_ids=positions)
        dist.send(forward_feature, dst=0)
        

########################## Edge Server ##########################
def train_server(args, model, dataloader, writer:SummaryWriter):
    # process exclusively occupy device
    torch.cuda.set_device(args.device_sv)

    # initialize data, model, gloabl_step, recv_tensor, and evaluator
    trainloader, testloader = dataloader
    forward_feature = torch.empty([args.batch_size,args.max_length,768]).to(args.device_sv)
    model.to(args.device_sv)
    global_step = 0
    total_loss = 0

    # vram = [torch.cuda.memory_allocated() / (1024 ** 2)]  # MB
    # print(f"VRAM: {vram} MB")

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, total_steps=args.num_steps)
    elif args.decay_type == "linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, total_steps=args.num_steps)
    else:
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    optimizer.zero_grad()

    train_fw_time = 0
    train_bw_time = 0

    model.train()
    trainloader = tqdm(trainloader, desc="Training (X / X Steps) (loss=X.X)")
    for batch in trainloader:
        labels, _ = batch
        labels = labels.to(args.device_sv)

        # Exchange feature and gradient
        dist.broadcast(forward_feature,1)
        forward_feature.requires_grad_(True)

        # fw_time = time.time()

        # Forward Server
        loss = run_forward(model, input_ids=forward_feature, label_ids=labels)
        total_loss += loss.item()

        # train_fw_time += time.time() - fw_time

        # if global_step < 2: vram.append( torch.cuda.memory_allocated() / (1024 ** 2))
        # vram = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        # print(f"after fw: {vram:.2f} MB")

        bw_time = time.time()
        # Backward Server
        backward_grad = run_backward(outputs=loss, input_tensor=forward_feature)

        train_bw_time += time.time() - bw_time

        # if global_step < 2: vram.append( torch.cuda.memory_allocated() / (1024 ** 2))
        # vram = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        # print(f"after bw: {vram:.2f} MB")

        # Exchange feature and gradient
        dist.broadcast(backward_grad,0)
        
        # Update Gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # if global_step < 2: vram.append( torch.cuda.memory_allocated() / (1024 ** 2))
        # vram = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        # print(f"after optim: {vram:.2f} MB")

        # Validation
        if global_step%args.num_interval == 0:
            args.global_step = global_step
            valid_server(args, model, testloader, writer)

        global_step += 1
        trainloader.set_description(
            "Training (%d / %d Steps) (loss=%2.5f) fw_time %2d:%2d, bw_time %2d:%2d" % (global_step, args.num_steps,
                                                                        total_loss/global_step,
                                                                        train_fw_time//60, train_fw_time%60,
                                                                        train_bw_time//60, train_bw_time%60)
        )

        # Update tensorboard
        writer.add_scalar("train/loss", scalar_value=total_loss/global_step, global_step=global_step)
        writer.add_scalar("train/lr_sv", scalar_value=optimizer.param_groups[0]['lr'], global_step=global_step)

        if global_step % args.num_steps == 0:
            break

        # if global_step == 2: 
        #     import matplotlib.pyplot as plt
        #     plt.plot(figsize=(5,5))
        #     plt.bar(range(len(vram)),vram)
        #     plt.show()

    writer.close()
    

def valid_server(args, model, testloader, writer:SummaryWriter):
    model.eval()
    forward_feature = torch.empty([args.batch_size,args.max_length,768]).to(args.device_sv)

    total_loss=0.0
    testloader = get_subset(testloader, args.num_test)
    data_iterator = tqdm(testloader)
    for step, batch in enumerate(data_iterator):
        labels, _ = batch
        labels = labels.to(args.device_sv)

        # forward_server
        dist.recv(forward_feature, src=1)
        loss = model(forward_feature, label_ids=labels)
        total_loss+=loss.item()

        data_iterator.set_description("Testing (%d / %d Steps) (loss=%2.5f)" % (step, len(testloader), loss.item()))
        
    writer.add_scalar("test/loss", scalar_value=total_loss/step, global_step=args.global_step)
    writer.add_scalar("test/perplexity", scalar_value=np.exp(total_loss/step), global_step=args.global_step)   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="all_news_GPT2_SplitLoRA")
    parser.add_argument("--text", type=str, default="WASHINGTON - The U.N. Security Council")
    parser.add_argument("--decay_type", choices=["cosine", "linear, const"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--num_split", type=int, default=2,
                        help="number of block id before split point")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=500,
                        help="total steps for fine-tuning")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_interval", type=int, default=50,
                        help="validation interval")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    args.device_sv = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device_cl = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # prepare datasets
    dataloader = get_loader(batch_size=args.batch_size, n_records=args.batch_size*args.num_steps/0.8, max_length=args.max_length)

    # prepare split models
    models = setup_gpt_models(args)

    # predict 64 words text
    pred_text(args.text, models)

    # train with multi-processes
    spawn(train, args=(args, models, dataloader), nprocs=args.world_size, join=True)

    #predict 64 words text
    pred_text(args.text, models)