# READTOKEN: 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


# Define Training Parameters
import argparse

parser = argparse.ArgumentParser(description="Argument parser for model training.")

# Add arguments
parser.add_argument("--name", type=str, default="lora")
parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
parser.add_argument("--total_record", type=int, default=1000, help="Total number of records.")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Name of the model to use.")
parser.add_argument("--ckpt_name", type=str, default="datasets/llama2_7b_4bits.pth", help="Path to the model checkpoint.")
parser.add_argument("--batch_size", type=int, default=3, help="batch size for training.")
parser.add_argument("--partition_id", type=int, default=30, help="partition ID for the training process.")
parser.add_argument("--seed", type=int, default=42, help="ramdom seed for sampling data.")
parser.add_argument("--lora_layer_type", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    help="List of LoRA layer types to use.")
parser.add_argument("--max_length", type=int, default=128, help="maximum sequence length.")
parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
parser.add_argument("--rank", type=float, default=4, help="lora rank")

args = parser.parse_args()
epochs = args.epochs
total_record = args.total_record
model_name = args.model_name
ckpt_name = args.ckpt_name
batch_size = args.batch_size
partition_id = args.partition_id
lora_layer_type = args.lora_layer_type
max_length = args.max_length
seed = args.seed

import torch
import random
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Get DataLoader Default: 80% data are used for training, 20% data are used for validation
from utils.data_utils import get_loader
train_dataloader = get_loader(batch_size=batch_size, n_records=total_record, max_length=128)["train"]
valid_dataloader = get_loader(batch_size=batch_size, n_records=total_record, max_length=128)["valid"]

# Setup Quantization Config
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Setup LoRA Config
from peft import LoraConfig
lora_config = LoraConfig(
    r=args.rank,  # Rank of the decomposition matrices
    lora_alpha=16,  # Scaling factor for the low-rank matrices
    target_modules=lora_layer_type,  # Target attention layers
    lora_dropout=0.05,  # Dropout applied to LoRA layers
    bias="none" # No bias matrices
)

# Setup Model Config
from utils.model_utils import ModelConfig
model_config = ModelConfig(
    model_name=model_name,
    state_dict=ckpt_name,
    partition_id=partition_id,
    total_step=epochs*len(train_dataloader),
    lora_config = lora_config,
    bnb_config = bnb_config,
)

import torch.distributed as dist
from utils.transmit_utils import send, recv
def runtime(rank, semaphore):
    args.semaphore = semaphore

    dist.init_process_group("nccl", rank=rank, world_size=2)

    if rank==0:
        model = thread_client(args, device="cuda:0", worker=1)
        state_dict = model.state_dict()
        # save model 0
        torch.save(state_dict,"out/model_0.pth")

    elif rank==1:
        model = thread_worker(args, device="cuda:1", client=0)
        # save model 1
        state_dict = model.state_dict()
        names = model.state_dict().keys()
        for name in names:
            if "layers" in name:
                replaced_name = "layer." + str(int(name.split(".")[1]) + partition_id) + "." + ".".join(map(str, name.split(".")[2:]))
                state_dict[replaced_name] = state_dict.pop(name)
        torch.save(state_dict,"out/model_1.pth")

    dist.destroy_process_group()

from torch import nn
def thread_client(args, device ="cuda:0", worker = 1)->nn.Module:
    # Parameter Setup
    from utils.model_utils import get_base_model_and_preprocess
    model_config.model_type="client"

    # Load model one at a time
    args.semaphore.acquire()   # Acquire the semaphore
    param = get_base_model_and_preprocess(config=model_config, device=device, lr=args.lr)
    args.semaphore.release()  # Release the semaphore

    # Initalize Parameters
    model = param["model"]
    optimizer = param["optimizer"]
    scheduler = param['scheduler']

    # Setup dowload_grad as transmission template
    download_grad = torch.empty([batch_size,max_length,model.hidden_size],dtype=torch.float16).to(device)

    # Synchornize all Devices
    args.semaphore.acquire()   # Acquire the semaphore
    args.semaphore.release()  # Release the semaphore  
    torch.cuda.synchronize(worker) # Synchornize with worker

    from utils.model_utils import run_backward, run_forward
    for epoch in range(epochs):
        # Training
        model.train()
        for input_ids, position_ids in train_dataloader:

            # Client FP
            output = run_forward(model, input_ids=input_ids.to(device), position_ids=position_ids.to(device))
            hidden_states = output["hidden_states"]            
            # position_embeddings = output["position_embeddings"]

            # Processing upload_feature and send to worker
            if input_ids.size(0) < download_grad.size(0):
                padded_hidden_states = torch.zeros_like(download_grad, dtype=upload_feature.dtype).to(device)
                padded_hidden_states[:input_ids.size(0),:] = hidden_states.detach()
                upload_feature = padded_hidden_states
            else:
                upload_feature = hidden_states.detach()
            dist.send(upload_feature, dst=worker)

            # Processing download_grad after recv from worker
            dist.recv(download_grad, src=worker)
            if input_ids.size(0) < download_grad.size(0):
                grad_tensor = download_grad[:input_ids.size(0),:].detach()
            else:
                grad_tensor = download_grad.detach()

            # Client BP
            run_backward(outputs = hidden_states, grad_tensor = grad_tensor, input_tensor = input_ids)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Evaluation
        model.eval()
        for input_ids, position_ids in valid_dataloader:
            with torch.no_grad():
                output = model.forward(input_ids.to(device), position_ids=position_ids.to(device))
                hidden_states = output["hidden_states"]
                # position_embeddings = output["position_embeddings"]

                # Processing upload_feature and send to worker
                if input_ids.size(0) < download_grad.size(0):
                    padded_hidden_states = torch.zeros_like(download_grad, dtype=upload_feature.dtype).to(device)
                    padded_hidden_states[:input_ids.size(0),:] = hidden_states.detach()
                    upload_feature = padded_hidden_states
                else:
                    upload_feature = hidden_states.detach()
                dist.send(upload_feature, dst=worker)
                

        torch.cuda.synchronize(worker)

    return model


def thread_worker(args, device ="cuda:1", client=0)->nn.Module:
    # Parameter Setup
    from utils.model_utils import get_base_model_and_preprocess
    model_config.model_type="worker"
    
    # Load model one at a time
    args.semaphore.acquire()   # Acquire the semaphore
    param = get_base_model_and_preprocess(config=model_config, device=device, lr=args.lr)
    args.semaphore.release()  # Release the semaphore

    # Initalize Parameters
    model = param["model"]
    optimizer = param["optimizer"]
    scheduler = param['scheduler']
    
    # Setup upload_feature as transmission template
    upload_feature = torch.empty([batch_size,max_length,model.hidden_size],dtype=torch.float16).to(device)

    # Synchornize all Devices
    args.semaphore.acquire()   # Acquire the semaphore
    args.semaphore.release()  # Release the semaphore
    torch.cuda.synchronize(client) # Synchornize with worker

    from tqdm import tqdm
    from utils.model_utils import run_backward, run_forward
    progress_bar = tqdm(range(epochs*len(train_dataloader)))
    for epoch in range(epochs):
        # Training
        model.train()
        for input_ids, position_ids in train_dataloader:

            # Processing upload_feature after recv from client
            dist.recv(upload_feature, src=client)
            if input_ids.size(0) < upload_feature.size(0):
                hidden_states = upload_feature[:input_ids.size(0),:].detach().requires_grad_(True)
            else:
                hidden_states = upload_feature.detach().requires_grad_(True)

            # Worker FP
            output = run_forward(model, input_ids=hidden_states, position_ids=position_ids.to(device), label_ids=input_ids.to(device))
            # Worker BP
            grad_tensor = run_backward(outputs=output["loss"], input_tensor=hidden_states)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Processing download_grad and send to client
            if input_ids.size(0) < upload_feature.size(0):
                padded_grad_tensor = torch.zeros_like(upload_feature, dtype=upload_feature.dtype)
                padded_grad_tensor[:input_ids.size(0),:] = grad_tensor.detach()
                download_grad = padded_grad_tensor
            else:
                download_grad = grad_tensor.detach()
            dist.send(download_grad, dst=client)

            # Update postfix
            progress_bar.set_description(f"Training Loss: {output["loss"].item():.5f}")
            progress_bar.update(1)

            with open("runtime_score_for_{args.name}.txt", "+a") as f:
                f.write(f"cycle:{progress_bar.n}, loss: {output["loss"].item():.5f} perplexity {torch.exp(output["loss"]):.0f}\n")
        
        with open("final_score_for_{args.name}.txt", "+a") as f:
            f.write(f"total_time: {progress_bar.format_dict['elapsed']:.12f} partition_id: {partition_id} lora_type: {lora_layer_type} ")

        # Evaluation
        eval_loss = 0
        model.eval()
        for input_ids, position_ids in valid_dataloader:
            with torch.no_grad():
                # Processing upload_feature after recv from client
                dist.recv(upload_feature, src=client)
                if input_ids.size(0) < upload_feature.size(0):
                    hidden_states = upload_feature[:input_ids.size(0),:].detach().requires_grad_(True)
                else:
                    hidden_states = upload_feature.detach().requires_grad_(True)

                output = run_forward(model, input_ids=hidden_states, position_ids=position_ids.to(device), label_ids=input_ids.to(device))
                eval_loss += output["loss"].item()

        print(f"Epoch {epoch+1} - Evaluation Loss: {eval_loss / len(valid_dataloader):.15f}")
    
        with open("final_score_for_{args.name}.txt", "+a") as f:
            f.write(f"eval loss: {eval_loss / len(valid_dataloader)} \n")
    
        torch.cuda.synchronize(client)

    return model


import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

if __name__ == '__main__':

    import os
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["PYTHONHASHSEED"] = str(seed)

    from multiprocessing import Manager
    with Manager() as manager:
        # Semaphore allowing a maximum of 2 concurrent processes
        shared_semaphore = manager.Semaphore(1)

        from torch.multiprocessing import spawn
        spawn(runtime, args=(shared_semaphore,), nprocs=2, join=True)