# import torch

# state_dict = torch.load("./datasets/lm_head_default.pth")

# print(state_dict["weight"].shape)

import torch

if torch.cuda.is_available():
    print("Available CUDA devices:")
    for i in range(torch.cuda.device_count()):
        print(f"Device ID: {i}, Name: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")

import os

# os.system(f"python deepseek_multipro.py --name {"qlora"} --total_record {1000} --rank {8} --batch_size {2} --partition_id {31} --lora_layer_type {"q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"}")
# os.system(f"python deepseek_multipro.py --name {"melora"} --total_record {1000} --rank {4} --batch_size {2} --partition_id {31} --lora_layer_type {"q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"}")
# os.system(f"python deepseek_multipro.py --name {"dlora"} --total_record {1000} --rank {8} --batch_size {3} --partition_id {31} --lora_layer_type {"q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"}")
os.system(f"python /home/edge02/Documents/SplitLoRA-main/DeepSeekLLM/deepseek_pipeline.py --name {"pipelora"} --total_record {1000} --rank {8} --batch_size {2} --partition_id {31} --lora_layer_type {"q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"}")
# /home/edge02/Documents/SplitLoRA-main/DeepSeekLLM/deepseek_multipro.py 

# os.system(f"python /home/edge02/Documents/SplitLoRA-main/DeepSeekLLM/deepseek_pipeline.py --partition_id {21}")
# os.system(f"python /home/edge02/Documents/SplitLoRA-main/DeepSeekLLM/deepseek_pipeline.py --partition_id {22}")
# os.system(f"python /home/edge02/Documents/SplitLoRA-main/DeepSeekLLM/deepseek_pipeline.py --partition_id {23}")
# os.system(f"python /home/edge02/Documents/SplitLoRA-main/DeepSeekLLM/deepseek_pipeline.py --partition_id {24}")
# os.system(f"python /home/edge02/Documents/SplitLoRA-main/DeepSeekLLM/deepseek_pipeline.py --partition_id 25")
# os.system(f"python /home/edge02/Documents/SplitLoRA-main/DeepSeekLLM/deepseek_pipeline.py --partition_id 26")
# os.system(f"python /home/edge02/Documents/SplitLoRA-main/DeepSeekLLM/deepseek_pipeline.py --partition_id 27")
# os.system(f"python /home/edge02/Documents/SplitLoRA-main/DeepSeekLLM/deepseek_pipeline.py --partition_id 28")
# os.sstem(f"python /home/edge02/Documents/SplitLoRA-main/DeepSeekLLM/deepseek_pipeline.py --partition_id 29")


# lora_case = [
#     # "q_proj k_proj",
#     # "gate_proj up_proj down_proj",
#     "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
#     ]

# # for lora_layer in lora_case: 
# # lora_layer = str(lora_layer).lstrip("[").rstrip("]")
for partition_id in range(20,30): 
    os.system(f"python /home/edge02/Documents/SplitLoRA-main/DeepSeekLLM/deepseek_pipeline.py --name {"pipelora"} --total_record {1000} --rank {8} --batch_size {2} --partition_id {partition_id} --lora_layer_type {"q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"}")