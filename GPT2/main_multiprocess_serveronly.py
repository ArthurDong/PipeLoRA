import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import spawn
import torch.distributed as dist

from models.GPT2 import GPT2LMHeadModel
from models.Lora import LoRA_GPT2
from utils.data_utils import get_config, get_weight, get_loader, get_subset, text_encode, text_decode
from utils.model_utils import run_forward, run_backward, model_split_transformer
from utils.scheduler import WarmupCosineSchedule, WarmupConstantSchedule, WarmupLinearSchedule


import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def print_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
    print(f"Allocated GPU memory: {allocated_memory:.2f} MB")
    print(f"Reserved GPU memory: {reserved_memory:.2f} MB")
    

def setup_gpt_models(args):
    # initialize GPT2 model
    model = GPT2LMHeadModel(get_config('./datasets/gpt2_config.json'))
    model = get_weight(model, './datasets/gpt2_pretrained.pth')

    # convert GPT model to LoRA-GPT
    LoRA_GPT2(gpt_model=model, r=4, alpha=8)

    # return LoRA GPT Model
    return {'model_cl':model,'model_sv':model}

def pred_text(pred_text, models):
    models['model_sv'].eval()

    input_ids, _ = text_encode(pred_text)
    input_ids = input_ids.unsqueeze(0).to(args.device_cl)
    models['model_sv'].to(args.device_cl)

    pred_ids = input_ids
    past = None
    for _ in range(args.max_length):

        logits, past = models['model_sv'](input_ids = input_ids, past = past)
        last_token_logits = logits[:, -1, :] / args.temperature

        # select topk values as predictions
        indices_to_remove = last_token_logits < torch.topk(last_token_logits, args.top_k)[0][:,-1]
        last_token_logits[indices_to_remove] = -1e10
    
        # get next text token
        probs = torch.nn.functional.softmax(last_token_logits, dim=-1) 
        input_ids = torch.multinomial(probs, num_samples=1)

        pred_ids = torch.cat([pred_ids, input_ids], dim=-1)

    print("Text Predicted:", text_decode(pred_ids))

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

    # initialize data and global_step, 

    trainloader, testloader = dataloader
    global_step = 0

    for batch in trainloader:

        inputs, positions = batch
        inputs = inputs.to(args.device_cl)
        positions = positions.to(args.device_cl)

        # Send input_ids
        dist.broadcast(inputs,1)

        # Send position_ids
        dist.broadcast(positions,1)

        # # Validation
        # if global_step%args.num_interval == 0:
        #     valid_client(args, model, testloader, writer)

        global_step += 1
        if global_step % args.num_steps == 0:
            break

    writer.close()


def valid_client(args, model, testloader, writer:SummaryWriter):
    
    testloader = get_subset(testloader, args.num_test)

    for batch in testloader:
        inputs, positions = batch
        inputs = inputs.to(args.device_cl)
        positions = positions.to(args.device_cl)

        # Send input_ids and position_ids
        dist.send(inputs, dst=0)
        dist.send(positions, dst=0)
        

########################## Edge Server ##########################
def train_server(args, model, dataloader, writer:SummaryWriter):
    # process exclusively occupy device
    torch.cuda.set_device(args.device_sv)

    print("Memory before loading the model:")
    print_gpu_memory()

    # initialize data, model, gloabl_step, recv_tensor, and evaluator
    trainloader, testloader = dataloader
    inputs = torch.empty([args.batch_size,1024],dtype=torch.long).to(args.device_sv)
    positions = torch.empty([args.batch_size,1024],dtype=torch.long).to(args.device_sv)
    model.to(args.device_sv)
    global_step = 0
    total_loss = 0

    print("Memory after loading the model:")
    print_gpu_memory()

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, total_steps=args.num_steps)
    elif args.decay_type == "linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, total_steps=args.num_steps)
    else:
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    optimizer.zero_grad()

    model.train()
    trainloader = tqdm(trainloader, desc="Training (X / X Steps) (loss=X.X)")
    for batch in trainloader:
        labels, _ = batch
        labels = labels.to(args.device_sv)

        # Receive input_ids
        dist.broadcast(inputs,1)

        # Receive position_ids
        dist.broadcast(positions,1)

        # Forward Server
        loss = run_forward(model, input_ids=inputs, position_ids=positions, label_ids=labels)
        total_loss += loss.item()

        # Backward Server
        run_backward(outputs=loss, input_tensor=inputs)
        
        # Update Gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # # Validation
        # if global_step%args.num_interval == 0:
        #     args.global_step = global_step
        #     valid_server(args, model, testloader, writer)

        global_step += 1
        trainloader.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, args.num_steps, total_loss/global_step)
        )

        # Update tensorboard
        writer.add_scalar("train/loss", scalar_value=total_loss/global_step, global_step=global_step)
        writer.add_scalar("train/lr_sv", scalar_value=optimizer.param_groups[0]['lr'], global_step=global_step)

        if global_step % args.num_steps == 0:
            break

    writer.close()
    

def valid_server(args, model, testloader, writer:SummaryWriter):
    model.eval()
    inputs = torch.empty([args.batch_size,1024],dtype=torch.long).to(args.device_sv)
    positions = torch.empty([args.batch_size,1024],dtype=torch.long).to(args.device_sv)

    total_loss=0.0
    testloader = get_subset(testloader, args.num_test)
    data_iterator = tqdm(testloader)
    for step, batch in enumerate(data_iterator):
        labels, _ = batch
        labels = labels.to(args.device_sv)

        # Receive input_ids and position_ids
        dist.recv(inputs, src=1)
        dist.recv(positions, src=1)
        loss = model(inputs, positions, label_ids=labels)
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
    parser.add_argument("--num_split", type=int, default=1,
                        help="number of block id before split point")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=2000,
                        help="total steps for fine-tuning")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_interval", type=int, default=200,
                        help="validation interval")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    args.device_sv = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device_cl = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # prepare datasets
    dataloader = get_loader(batch_size=args.batch_size, n_records=args.batch_size*args.num_steps/0.8)

    # prepare split models
    models = setup_gpt_models(args)

    #predict text
    pred_text(args.text, models)

    # train with multi-processes
    spawn(train, args=(args, models, dataloader), nprocs=args.world_size, join=True)

    #predict text
    pred_text(args.text, models)