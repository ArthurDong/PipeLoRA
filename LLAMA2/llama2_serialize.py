# READTOKEN: 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


# Define Training Parameters
epochs = 1
total_record = 1000
model_name = "meta-llama/Llama-2-7b-hf"
ckpt_name = "./datasets/llama2_7b_4bits.pth"
batch_size = 2
partition_id = 30
seed = 42
lora_layer_type = ["q_proj", "v_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lr = 2e-4

import torch
import random
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
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
    r=4,  # Rank of the decomposition matrices
    lora_alpha=16,  # Scaling factor for the low-rank matrices
    target_modules=lora_layer_type,  # Target attention layers
    lora_dropout=0.00,  # Dropout applied to LoRA layers
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

# Parameter Setup
from utils.model_utils import get_base_model_and_preprocess
model_config.model_type="client"
param_0 = get_base_model_and_preprocess(config=model_config, device="cuda:0", lr=lr)
model_config.model_type="worker"
param_1 = get_base_model_and_preprocess(config=model_config, device="cuda:1", lr=lr)

model_0 = param_0["model"]
optimizer_0 = param_0["optimizer"]
scheduler_0 = param_0['scheduler']

model_1 = param_1["model"]
optimizer_1 = param_1["optimizer"]
scheduler_1 = param_1['scheduler']

from tqdm import tqdm
from utils.model_utils import run_backward, run_forward
progress_bar = tqdm(range(epochs*len(train_dataloader)))
for epoch in range(epochs):
    # Training
    model_0.train()
    model_1.train()
    for input_ids, position_ids in train_dataloader:

        mid_output = run_forward(model_0, input_ids=input_ids.to("cuda:0"), position_ids=position_ids.to("cuda:0"))
        hidden_states_input=mid_output["hidden_states"].detach()
        # position_embeddings=mid_output["position_embeddings"]

        hidden_states_input.requires_grad_(True)
        output = run_forward(model_1, input_ids=hidden_states_input.to("cuda:1"), position_ids=position_ids.to("cuda:1"), label_ids=input_ids.to("cuda:1"))
        grad_tensor = run_backward(outputs=output["loss"], input_tensor=hidden_states_input)

        print("\n hidden_states",hidden_states_input[0,:5,:5],
              "\n logits:",output["logits"][0,:5,:5])
        print("grad_tensor",grad_tensor[0,0,1:5])

        optimizer_1.step()
        scheduler_1.step()
        optimizer_1.zero_grad()
        
        run_backward(outputs = mid_output["hidden_states"], grad_tensor = grad_tensor.detach(), input_tensor = input_ids)
        
        optimizer_0.step()
        scheduler_0.step()
        optimizer_0.zero_grad()
       

        # Update postfix
        progress_bar.set_description(f"Training Loss: {output["loss"].item():.5f}")
        progress_bar.update(1)

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

# Inference
input_text = "NEW YORK (Reuters) - U.S. "
from utils.data_utils import pred_text_llama2
print(pred_text_llama2(prompt_text=input_text, model=[model_0, model_1]))