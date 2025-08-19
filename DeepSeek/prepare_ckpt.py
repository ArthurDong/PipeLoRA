import torch
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig

# Path to your checkpoint and model configuration
model_name = "deepseek-ai--deepseek-llm-7b-base"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# model = LlamaModel.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda:0")
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda:0")

# Save current state_dict to external file
state_dict = model.state_dict()

names = model.state_dict().keys()
for name in names:
    if "model" in name:
        replaced_name = ".".join(map(str, name.split(".")[1:]))
        state_dict[replaced_name] = state_dict.pop(name)
        print(f"replace key:{name} with {replaced_name}")

torch.save(state_dict,"./datasets/deepseek_7b_4bits.pth")
torch.save({'weight':state_dict["lm_head.weight"]},"./datasets/lm_head_default.pth")

# Load state_dict from external file
# state_dict = torch.load("./datasets/llama2_7b_4bits.pth", weights_only=True)
# model.load_state_dict(state_dict, strict=False)

input_text = "NEW YORK (Reuters) - U.S. "
from utils.data_utils import pred_text_deepseek
print(pred_text_deepseek(prompt_text=input_text, model=model))