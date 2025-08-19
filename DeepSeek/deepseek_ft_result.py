# Inference
import torch

# Define Training Parameters
model_name = "deepseek-ai/deepseek-llm-7b-base"
lora_layer_type = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
device = "cuda:0" if torch.cuda.is_available() else "cpu"


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
    r=8,  # Rank of the decomposition matrices
    lora_alpha=16,  # Scaling factor for the low-rank matrices
    target_modules=lora_layer_type,  # Target attention layers
    lora_dropout=0.05,  # Dropout applied to LoRA layers
)

from models.DeepSeek import DeepSeekModel
from peft import get_peft_model
from utils.model_utils import Full_Model
# Setup Llama2 Quantization models 
model = DeepSeekModel.from_pretrained(model_name, quantization_config=bnb_config, device_map=device)
# Setup Llama2 LoRA
model = get_peft_model(model, lora_config)
# Setup Llama2 Full
model = Full_Model(model)

# Load state_dict to models
# model_0_state_dict = torch.load("./out/model_0.pth", weights_only=True)
# model_1_state_dict = torch.load("./out/model_1.pth", weights_only=True)
state_dict = torch.load("./datasets/deepseek_7b_4bits.pth", weights_only=True)

model.load_state_dict(state_dict, strict=False)
# model.load_state_dict(model_0_state_dict, strict=False)
# model.load_state_dict(model_1_state_dict, strict=False)

# Inference
input_text = "NEW YORK (Reuters) - U.S. "
from utils.data_utils import pred_text_deepseek
print(pred_text_deepseek(prompt_text=input_text, model=model))