# READTOKEN: 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

# Define Training Parameters
import argparse
parser = argparse.ArgumentParser(description="Argument parser for model training.")
import math
# Add arguments
parser.add_argument("--name", type=str, default="all_news_Llama2_SplitLoRA")
parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")############################
parser.add_argument("--total_record", type=int, default=3750, help="Total number of records.")########################
parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-llm-7b-base", help="Name of the model to use.")
parser.add_argument("--ckpt_name", type=str, default="./datasets/deepseek_7b_4bits.pth", help="Path to the model checkpoint.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")################################
parser.add_argument("--partition_id", type=int, default=1, help="Partition ID for the training process.")#############
parser.add_argument("--seed", type=int, default=42, help="Ramdom seed for sampling data.")
parser.add_argument("--lora_layer_type", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    help="List of LoRA layer types to use.")#####################################
# parser.add_argument("--lora_layer_type", nargs="+", default=["q_proj", "v_proj", "k_proj", "o_proj", ],
                    # help="List of LoRA layer types to use.")#####################################
parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length.")
##
parser.add_argument("--device_delay", nargs="+", type=float, default=[0.0, 0.0],
                    help="Delay time for each device (in seconds).")
##
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
device_delay = args.device_delay

import torch
import random
torch.manual_seed(seed)
random.seed(seed)

# Get DataLoader Default: 80% data are used for training, 20% data are used for validation
from utils.data_utils import get_loader
train_dataloader = get_loader(batch_size=batch_size, n_records=total_record, max_length=128)["train"]
valid_dataloader = get_loader(batch_size=batch_size, n_records=total_record, max_length=128)["valid"]

###########################################################################################
# Setup Quantization Config
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
############################################################################################

# Setup LoRA Config
from peft import LoraConfig
lora_config = LoraConfig(
    r=16,  # Rank of the decomposition matrices
    lora_alpha=8,  # Scaling factor for the low-rank matrices
    target_modules=lora_layer_type,  # Target attention layers
    lora_dropout=0.05,  # Dropout applied to LoRA layers
)

# Setup Model Config
from utils.model_utils import ModelConfig
model_config = ModelConfig(
    model_name=model_name,
    state_dict=ckpt_name,
    partition_id=partition_id,
    total_step=epochs*len(train_dataloader),
    lora_config = lora_config,
    #################################################################
    #bnb_config = bnb_config,
    ################################################################
)

import torch.distributed as dist
def runtime(rank, semaphore):
    args.semaphore = semaphore

    dist.init_process_group("nccl",init_method="env://", rank=rank, world_size=2)

    if rank==0:
        model = thread_client(args,device="cuda:0")
        state_dict = model.state_dict()
        # save model 0
        torch.save(state_dict,"out/model_0.pth")

    elif rank==1:
        model = thread_server(args,device="cuda:1")
        # save model 1
        state_dict = model.state_dict()
        names = model.state_dict().keys()
        for name in names:
###################################################################
             # 跳过 thop 添加的额外属性
            if name in ["total_ops", "total_params"]:
                continue
            if "layers" in name:
                try:
                    # 尝试解析层编号
                    layer_id = int(name.split(".")[1])
                    #replaced_name = "layer." + str(int(name.split(".")[1]) + partition_id) + "." + ".".join(map(str, name.split(".")[2:]))
                    replaced_name = "layer." + str(layer_id + partition_id) + "." + ".".join(map(str, name.split(".")[2:]))
                    state_dict[replaced_name] = state_dict.pop(name)
                except (ValueError, IndexError):
                    # 如果解析失败，跳过该键
                    continue
###################################################################
        torch.save(state_dict,"out/model_1.pth")
   
    dist.destroy_process_group()

from torch import nn
##
import time
import datetime  
## 
#######################################################
def log_gpu_memory(device="cuda:1"):
    # 设置当前设备
    torch.cuda.set_device(device)
    if torch.cuda.is_available():
        # 获取当前时间戳
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        allocated = torch.cuda.memory_allocated() / 1024**2  # 当前显存占用（MB）
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # 峰值显存占用（MB）
        cached = torch.cuda.memory_reserved() / 1024**2  # 显存缓存占用（MB）
        
        log_str = (
            f"[{timestamp}] [memory monitoring]\n"
            f"Current memory usage: {allocated:.2f} MB\n"
            f"Peak memory usage: {max_allocated:.2f} MB\n"
            f"memory cache occupancy: {cached:.2f} MB\n"
            "--------------------------\n"
        )
        
        with open("Train_time_for_multipro.txt", "a") as f:
            f.write(log_str)
        # # 重置峰值显存统计
        # torch.cuda.reset_max_memory_allocated()
    else:
        print("CUDA not work\n")

from thop import profile
from copy import deepcopy

def log_model_flops(model, hidden_states, device="cuda:1"):

    model_copy = deepcopy(model)
    # 构造虚拟的 label_ids（假设词汇量是 32000）
    #dummy_label_ids = torch.randint(0, 32000, (batch_size, max_length)).to(device)
    dummy_label_ids = torch.randint(0, 32000, (batch_size, max_length), dtype=torch.long).to(device)
    # 计算 FLOPs
    flops, params = profile(model_copy, inputs=(hidden_states, dummy_label_ids),verbose=False)
    #写入日志
    log_str = (
        f"[FLOPs statistics]\n"
        f"theoretical FLOPs: {flops / 1e9:.2f} GFLOPs\n"
        f"Number of parameters: {params / 1e6:.2f} Million\n"
        "--------------------------\n"
    )

    with open("Train_time_for_multipro.txt", "a") as f:
        f.write(log_str)

#######################################################
def thread_client(args, device ="cuda:0")->nn.Module:
    # Parameter Setup
    from utils.model_utils import get_base_model_and_preprocess
    model_config.model_type="client"

    # Load model one at a time
   
    args.semaphore.acquire()   # Acquire the semaphore
    param = get_base_model_and_preprocess(config=model_config, device=device)
    args.semaphore.release()  # Release the semaphore
    
    # Initalize Parameters
    model = param["model"].to(device)
    optimizer = param["optimizer"]
    scheduler = param['scheduler']
    grad_tensor = torch.empty([batch_size,128,4096],dtype=torch.float32).to(device)

    # Synchornize all Devices
    args.semaphore.acquire()   # Acquire the semaphore
    args.semaphore.release()  # Release the semaphore  

    #########################################################################
     # 获取当前设备的延迟时间
    delay_time = args.device_delay[0]  # 获取对应的延迟时间

    #在关键操作前添加延迟
    time.sleep(delay_time)
    ##########################################################################

    from utils.model_utils import run_backward, run_forward
    for epoch in range(epochs):
        # Training
        model.train()
        for input_ids, position_ids in train_dataloader:

            output = run_forward(model, input_ids=input_ids.to(device), position_ids=position_ids.to(device))
            hidden_states = output["hidden_states"]
            position_embeddings = output["position_embeddings"]

            
            # Exchange feature and gradient
            dist.send(hidden_states, dst=1,)
            # Exchange feature and gradient
            dist.recv(grad_tensor, src=1,)

            run_backward(outputs = hidden_states, grad_tensor = grad_tensor, input_tensor = input_ids,model=model)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Evaluation
    ######################################################
        torch.cuda.empty_cache()
    ######################################################
        model.eval()
        for input_ids, position_ids in valid_dataloader:
            with torch.no_grad():
                output = model.forward(input_ids.to(device), position_ids=position_ids.to(device))
                hidden_states = output["hidden_states"]
                position_embeddings = output["position_embeddings"]
                dist.send(hidden_states, dst=1)

    return model

import time
def thread_server(args, device ="cuda:1")->nn.Module:
    # Parameter Setup
    from utils.model_utils import get_base_model_and_preprocess
    model_config.model_type="server"
    
    # Load model one at a time
   
    args.semaphore.acquire()   # Acquire the semaphore
    param = get_base_model_and_preprocess(config=model_config, device=device)
    args.semaphore.release()  # Release the semaphore
    
    # Initalize Parameters
    model = param["model"].to(device)
    optimizer = param["optimizer"]
    scheduler = param['scheduler']
    hidden_states = torch.empty([batch_size,128,4096],dtype=torch.float32).to(device)

    # Synchornize all Devices
    args.semaphore.acquire()   # Acquire the semaphore
    args.semaphore.release()  # Release the semaphore

    #########################################################################
    # 获取当前设备的延迟时间
    delay_time = args.device_delay[1]  # 获取对应的延迟时间

    # 在关键操作前添加延迟
    time.sleep(delay_time)
    ############################################################################

    from tqdm import tqdm
    from utils.model_utils import run_backward, run_forward
    progress_bar = tqdm(range(epochs*len(train_dataloader)))


    for epoch in range(epochs):
##################################################################################
    # Initialize time tracking
        total_batches = 0
        total_samples = 0
        start_time = time.perf_counter()  # Start the timer for throughput calculation
####################################################################################
        # Training
        model.train()
        for input_ids, position_ids in train_dataloader:

            # Process each batch
            total_batches += 1
            total_samples += input_ids.size(0)  # Accumulate the total number of samples

            # Exchange feature and gradient
            dist.recv(hidden_states, src=0)
            hidden_states=hidden_states.detach().requires_grad_(True)
            output = run_forward(model, input_ids=hidden_states.to(device), position_ids=position_ids.to(device), label_ids=input_ids.to(device))
            grad_tensor = run_backward(outputs=output["loss"], input_tensor=hidden_states,model=model)

            #print(f"Rank {progress_bar}: Sending data to {}")
            # Exchange feature and gradient
            dist.send(grad_tensor, dst=0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update postfix
            progress_bar.set_description(f"Training Loss: {output['loss'].item():.5f}")
            progress_bar.update(1)

        ####################################################################################
        # 计算 FLOPs（传递 hidden_states 和虚拟 label_ids）
        log_model_flops(model, hidden_states, device=device)  # 添加 device 参数

        # 记录显存占用
        log_gpu_memory(device=device)
         #####################################################################################
        
        with open("Train_time_for_multipro.txt", "+a") as f:
            f.write(f"total_time: {progress_bar.format_dict['elapsed']:.12f} partition_id: {partition_id} lora_type: {lora_layer_type} \n")

########################################################################################################
        # End of epoch: Calculate and log throughput
        elapsed_time = time.perf_counter() - start_time  # Calculate elapsed time for the epoch
        throughput_batches = total_batches / elapsed_time  # Batches per second
        throughput_samples = total_samples / elapsed_time  # Samples per second
#########################################################################################################
        
        with open("Train_time_for_multipro.txt", "+a") as f:
            f.write(f"Epoch {epoch + 1} - Throughput: {throughput_samples:.2f} samples/sec, {throughput_batches:.2f} batches/sec\n")
        # Evaluation
    ######################################################
        torch.cuda.empty_cache()
    ######################################################
        eval_loss = 0
        model.eval()
        for input_ids, position_ids in valid_dataloader:
            with torch.no_grad():
                dist.recv(hidden_states, src=0)
                # hidden_states.requires_grad_(True)
                output = run_forward(model, input_ids=hidden_states.to(device), position_ids=position_ids.to(device), label_ids=input_ids.to(device))
                eval_loss += output["loss"].item()
        
        # Calculate perplexity
        perplexity = math.exp(eval_loss / len(valid_dataloader))

        print(f"Epoch {epoch+1} - Evaluation Loss: {eval_loss / len(valid_dataloader):.15f}")
    
        with open("Train_time_for_multipro.txt", "+a") as f:
            f.write(f"loss: {eval_loss / len(valid_dataloader)} \n")
            f.write(f"perplexity: {perplexity} \n")
            f.write("#" * 50 + "\n")  # 打印一排 # 号，长度可以根据需要调整
    
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

    from multiprocessing import Manager
    with Manager() as manager:
        # Semaphore allowing a maximum of 2 concurrent processes
        shared_semaphore = manager.Semaphore(1)

        from torch.multiprocessing import spawn
        spawn(runtime, args=(shared_semaphore,), nprocs=2, join=True)