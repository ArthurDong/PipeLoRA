import torch
from old.LLAMA2 import LlamaForCausalLM
from models.LLAMA2 import LlamaModel
from transformers import BitsAndBytesConfig

# Path to your checkpoint and model configuration
model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# model = LlamaModel.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda:0")
model = LlamaForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda:0")

# Save current state_dict to external file
state_dict = model.state_dict()

names = model.state_dict().keys()
for name in names:
    if "model" in name:
        replaced_name = ".".join(map(str, name.split(".")[1:]))
        state_dict[replaced_name] = state_dict.pop(name)
        print(f"replace key:{name} with {replaced_name}")

torch.save(state_dict,"./datasets/llama2_7b_4bits.pth")

# Load state_dict from external file
# state_dict = torch.load("./datasets/llama2_7b_4bits.pth", weights_only=True)
# model.load_state_dict(state_dict, strict=False)

input_text = "NEW YORK (Reuters) - U.S. "
from utils.data_utils import pred_text_llama2
print(pred_text_llama2(prompt_text=input_text, model=model))