# READTOKEN: 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
# Define Training Parameters
epochs = 1
total_record = 3750
model_name = "deepseek-ai/deepseek-llm-7b-base"
ckpt_name = "./datasets/llama2_7b_4bits.pth"
batch_size = 1
partition_id = 31
lora_layer_type = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
max_length=128

import torch
import random
import math
seed = 42
torch.manual_seed(seed)
random.seed(seed)

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
    r=32,  # Rank of the decomposition matrices
    lora_alpha=8,  # Scaling factor for the low-rank matrices
    target_modules=lora_layer_type,  # Target attention layers
    lora_dropout=1e-5,  # Dropout applied to LoRA layers
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

from torch import nn
##
import time
import datetime  
## 
#######################################################
def log_gpu_memory(device="cuda:0"):
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
        
        with open("Train_time_for_serialize.txt", "a") as f:
            f.write(log_str)
        # # 重置峰值显存统计
        # torch.cuda.reset_max_memory_allocated()
    else:
        print("CUDA not work\n")
#######################################################
# Parameter Setup

from utils.model_utils import get_base_model_and_preprocess
model_config.model_type="client"
param_0 = get_base_model_and_preprocess(config=model_config, device="cuda:0")
model_config.model_type="server"
param_1 = get_base_model_and_preprocess(config=model_config, device="cuda:1")

model_0 = param_0["model"]
optimizer_0 = param_0["optimizer"]
scheduler_0 = param_0['scheduler']

model_1 = param_1["model"]
optimizer_1 = param_1["optimizer"]
scheduler_1 = param_1['scheduler']

from tqdm import tqdm
from utils.model_utils import run_backward, run_forward
progress_bar = tqdm(range(epochs*len(train_dataloader)))
cycle = 0  # Initialize cycle counter
for epoch in range(epochs):
    # Training
    model_0.train()
    model_1.train()
    for input_ids, position_ids in train_dataloader:
        cycle += 1  # Increment cycle number
        mid_output = run_forward(model_0, input_ids=input_ids.to("cuda:0"), position_ids=position_ids.to("cuda:0"))
        hidden_states_input=mid_output["hidden_states"].detach().to("cuda:0")
        position_embeddings=mid_output["position_embeddings"]

        hidden_states_input.requires_grad_(True)
        output = run_forward(model_1, input_ids=hidden_states_input.to("cuda:1"), position_ids=position_ids.to("cuda:1"), label_ids=input_ids.to("cuda:1"))
        
        grad_tensor = run_backward(outputs=output["loss"], input_tensor=hidden_states_input,model=model_0)
        run_backward(outputs = mid_output["hidden_states"], grad_tensor = grad_tensor, input_tensor = input_ids,model=model_1)

        optimizer_0.step()
        optimizer_1.step()
        scheduler_0.step()
        scheduler_1.step()
        optimizer_0.zero_grad()
        optimizer_1.zero_grad()

        eval_loss = output['loss'].item()  # Current cycle loss
        perplexity = math.exp(eval_loss)  # Calculate perplexity

        # Update postfix
        progress_bar.set_description(f"Training Loss: {output['loss'].item():.5f}")
        progress_bar.update(1)

        # Log the results for the current cycle
        with open("Train_time_for_serialize.txt", "+a") as f:
            f.write(f"Cycle {cycle} -- loss: {eval_loss}")
            f.write(f"-- perplexity: {perplexity} \n")
            f.write("#" * 50 + "\n")

    # 每个epoch打印 GPU 内存使用情况
    log_gpu_memory(device="cuda:0")  # 打印 GPU 0 的内存使用情况
    log_gpu_memory(device="cuda:1")  # 打印 GPU 1 的内存使用情况

    with open("Train_time_for_serialize.txt", "+a") as f:
            f.write(f"total_time: {progress_bar.format_dict['elapsed']:.12f} partition_id: {partition_id} lora_type: {lora_layer_type}\n ")


    # Evaluation
    eval_loss = 0
    model_0.eval()
    model_1.eval()
    for input_ids, position_ids in valid_dataloader:
        with torch.no_grad():
            mid_output = run_forward(model_0, input_ids.to("cuda:0"), position_ids=position_ids.to("cuda:0"))
            output = run_forward(model_1, mid_output["hidden_states"].to("cuda:1"), position_ids=position_ids.to("cuda:1"), label_ids=input_ids.to("cuda:1"))
            eval_loss += output["loss"].item()

    print(f"Epoch {epoch+1} - Evaluation Loss: {eval_loss / len(valid_dataloader)}")
    # with open("Train_time_for_serialize.txt", "+a") as f:
    #     f.write(f"loss: {eval_loss / len(valid_dataloader)} \n")
    #     f.write(f"perplexity: {perplexity} \n")
    #     f.write("#" * 50 + "\n")  # 打印一排 # 号，长度可以根据需要调整
# # Inference
# input_text = "NEW YORK (Reuters) - U.S. "
# from utils.data_utils import pred_text_llama2
# print(pred_text_llama2(prompt_text=input_text, model=[model_0, model_1]))