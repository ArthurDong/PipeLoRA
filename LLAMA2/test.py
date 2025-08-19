import torch

if torch.cuda.is_available():
    print("Available CUDA devices:")
    for i in range(torch.cuda.device_count()):
        print(f"Device ID: {i}, Name: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")

import os

lora_case = [
    "q_proj k_proj",
    "q_proj k_proj v_proj o_proj",
    "gate_proj up_proj down_proj",
    "q_proj k_proj gate_proj up_proj down_proj",
    "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
    ]

for lora_layer in lora_case: 
    lora_layer = str(lora_layer).lstrip("[").rstrip("]")
    for partition_id in range(1,31): 
        os.system(f"python llama2_pipeline.py --total_record {80} --partition_id {partition_id} --lora_layer_type {lora_layer}")

for lora_layer in lora_case: 
    lora_layer = str(lora_layer).lstrip("[").rstrip("]")
    for partition_id in range(1,31): 
        os.system(f"python llama2_multipro.py --total_record {80} --partition_id {partition_id} --lora_layer_type {lora_layer}")

# os.system(f"python llama2_multipro.py --total_record {80} --partition_id {30} ")
