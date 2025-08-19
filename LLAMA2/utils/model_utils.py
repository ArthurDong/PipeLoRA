import torch
import torch.nn as nn
import torch.distributed as dist

#########################################################
#                    model utils                        #
#########################################################

def run_forward(model:nn.Module, input_ids: torch.Tensor, position_ids:torch.Tensor = None, label_ids: torch.Tensor = None)->torch.Tensor:
    """
    Redefine forward pass
    Args: model, nn.Modules, running model
          inputs, Tensor, imput tensor
    Return: model.forward(inputs), Tensor
    """
    # define ctx hook
    def pack_hook(x):
        return x

    def unpack_hook(x):
        return x

    # Run forward pass
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        if label_ids is None:
            return model.forward(input_ids, position_ids=position_ids)
        else:
            return model.forward(input_ids, label_ids=label_ids)
    
def run_backward(outputs:torch.Tensor, grad_tensor:torch.Tensor=None, input_tensor:torch.Tensor=None)->torch.Tensor:
    """
    Redefine backward pass
    Args: outputs,  Tensor,  layer's output tensor
          grad_tensor, Tensor, next-layer's gradient 
          input_tensor, Tensor, layer's input tensor
          optimizer, Optimizer, optimizer
    Return: input_tensor.grad, Tensor, layer's input tensor's gradient
    """
    # Run backward pass
    torch.autograd.backward(outputs, grad_tensors=grad_tensor)
    
    # return input gradient for backward chain
    if input_tensor != None:
        # same as: input_tensor.register_hook(lambda grad: grad)
        return input_tensor.grad

#########################################################
#                   Llama2 models                       #
#########################################################

class ModelConfig:
    model_type=None

    def __init__(self, model_name, state_dict=None, lora_config=None, bnb_config=None, partition_id=0, total_step=0):
        self.model_name = model_name
        self.state_dict = state_dict
        self.partition_id = partition_id
        self.total_step = total_step
        self.lora_config = lora_config
        self.bnb_config = bnb_config

class Client_Model(nn.Module):
    def __init__(self, model:nn.Module, split_layer_id:int):
        super().__init__()

        self.embed_tokens = model.embed_tokens
        self.rotary_emb = model.rotary_emb
        self.layers = model.layers[:split_layer_id]
        self.device = model.device

    def forward(self, input_ids: torch.LongTensor = None, position_ids = None): 
        # wte 
        inputs_embeds = self.embed_tokens(input_ids)

        # wpe 
        if position_ids is None:
            position_ids = torch.arange(0, inputs_embeds.shape[1]).to(device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # transformers
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=None,
                    use_cache=None,
                    cache_position=None,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]
        
        return {"hidden_states":hidden_states, "position_embeddings":position_embeddings}

class Server_Model(nn.Module):
    def __init__(self, model:nn.Module, split_id:int):
        super().__init__()
        self.vocab_size =  model.vocab_size
        self.embed_tokens = model.embed_tokens
        self.rotary_emb = model.rotary_emb
        self.layers = model.layers[split_id:]
        self.norm = model.norm
        self.lm_head = model.lm_head
        self.device = model.device

    # def forward(self, hidden_states: torch.LongTensor = None, label_ids: torch.LongTensor = None):
    def forward(self, hidden_states: torch.LongTensor = None, label_ids: torch.LongTensor = None):
        # wte 
        inputs_embeds = self.embed_tokens(label_ids)

        # wpe 
        position_ids = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # transformers
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=None,
                    use_cache=None,
                    cache_position=None,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

        # norm
        hidden_states = self.norm(hidden_states)

        # pred
        logits = self.lm_head(hidden_states).float()

        loss = None
        if label_ids is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return {"loss":loss, "logits":logits}

class Full_Model(nn.Module):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.vocab_size =  model.vocab_size
        self.embed_tokens = model.embed_tokens
        self.rotary_emb = model.rotary_emb
        self.layers = model.layers
        self.norm = model.norm
        self.lm_head = model.lm_head
        self.device = model.device

    # def forward(self, hidden_states: torch.LongTensor = None, label_ids: torch.LongTensor = None):
    def forward(self, input_ids: torch.LongTensor = None, position_ids = None, label_ids: torch.LongTensor = None):
       
        # wte 
        inputs_embeds = self.embed_tokens(input_ids)

        # wpe 
        if position_ids is None:
            position_ids = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # transformers
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=None,
                    use_cache=None,
                    cache_position=None,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

        # norm
        hidden_states = self.norm(hidden_states)

        # pred
        logits = self.lm_head(hidden_states).float()

        loss = None
        if label_ids is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return {"loss":loss, "logits":logits}

from models.LLAMA2 import LlamaModel
from peft import get_peft_model
def get_base_model_and_preprocess(config, device):

    # Setup Llama2 Quantization models 
    model = LlamaModel.from_pretrained(config.model_name, quantization_config=config.bnb_config, device_map=device, )

    # Load Llama2 state_dict
    if isinstance(config.state_dict, str):
        state_dict = torch.load(config.state_dict, weights_only=True)
        # print(f"load ckeckpoints for device: {device}")
    model.load_state_dict(state_dict, strict=False)

    # Setup Llama2 LoRA
    model = get_peft_model(model, config.lora_config)

    # Setup Llama2 Partition models
    if config.model_type == "client":
        model = Client_Model(model, config.partition_id)
    elif config.model_type == "server":
        model = Server_Model(model, config.partition_id)
    else:
        print("Must Choose a model_type from either 'client' or 'server'!")
        exit(-1)

    del state_dict
    torch.cuda.empty_cache()

    # Define Optimizer
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    # Define Scheduler
    from transformers import get_scheduler
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1*config.total_step), num_training_steps=config.total_step)

    return {
        "model" : model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }