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
#                    GPT2 models                        #
#########################################################

def model_split_transformer(base_model:nn.Module, split_block_id:int)->dict:
    """ 
    base_model: model to be splited,
    split_block_id: split location id,
    models: return models
    """
    models = {}
    models['model_cl'] = LoRA_GPT_Client(base_model, split_block_id)
    models['model_sv'] = LoRA_GPT_Server(base_model, split_block_id)
    return models

class LoRA_GPT_Client(nn.Module):
    def __init__(self, model:nn.Module, split_id:int):
        super(LoRA_GPT_Client, self).__init__()
        self.wte = model.transformer.wte
        self.wpe = model.transformer.wpe
        self.split_h = nn.ModuleList()
        if split_id>0:
            for i in range(split_id):
                self.split_h.add_module(str(i), model.transformer.h[i])

    def forward(self, input_ids, position_ids=None, past=None):
        # process past
        if past is None:
            past_length = 0 
            past = [None] * len(self.split_h)
        else:
            past_length = past[0].size(-2)
        
        # process position
        if position_ids is None:
            position_ids = torch.arange(past_length, past_length + input_ids.size(-1), 
                                        dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # hedden_state = text embeddings and position embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # pass data through transformer blocks
        presents = []
        for block, layer_past in zip(self.split_h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)

        if presents == []:
            presents.append(position_ids.unsqueeze(-1))

        return hidden_states, presents

class LoRA_GPT_Server(nn.Module):
    def __init__(self, model, split_id):
        super(LoRA_GPT_Server, self).__init__()
        self.split_h = nn.ModuleList()
        if split_id<len(model.transformer.h):
            for i in range(split_id, len(model.transformer.h)):
                self.split_h.add_module(str(i), model.transformer.h[i])

        self.ln_f = model.transformer.ln_f
        self.decoder = model.head

    def forward(self, hidden_states, past=None, label_ids=None):
        # process past
        if past is None:
            past = [None] * len(self.split_h)

        # iterate through transformers
        presents = []
        try:
            for block, layer_past in zip(self.split_h, past):
                hidden_states, present = block(hidden_states, layer_past)
                presents.append(present)
        except:
            pass
    
        # pass through layer_norm + decoder
        hidden_states = self.ln_f(hidden_states)
        logits = self.decoder(hidden_states)

        # calculate loss if has label, else return predict logits
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, logits.size(-1)), label_ids.view(-1))
            return loss
        return logits, presents


class LoRA_GPT_Split_Combine(nn.Module):
    def __init__(
        self,
        LoRA_GPT_Client,
        LoRA_GPT_Server,
    ):
        super().__init__()
        self.LoRA_GPT_Client = LoRA_GPT_Client
        self.LoRA_GPT_Server = LoRA_GPT_Server

    def forward(self, input_ids, position_ids, label_ids):
        outputs_cl, _ = self.LoRA_GPT_Client(input_ids, position_ids)
        loss = self.LoRA_GPT_Server(outputs_cl, label_ids=label_ids)
        return loss

#########################################################
#                   Llama2 models                       #
#########################################################

def model_split_transformer(base_model:nn.Module, split_block_id:int)->dict:
    """ 
    base_model: model to be splited,
    split_block_id: split location id,
    models: return models
    """
    models = {}
    models['model_cl'] = LoRA_GPT_Client(base_model, split_block_id)
    models['model_sv'] = LoRA_GPT_Server(base_model, split_block_id)
    return models

from transformers.cache_utils import Cache, DynamicCache

class LoRA_Llama_Client(nn.Module):
    def __init__(self, model:nn.Module, split_id:int):
        super().__init__()

        self.embed_tokens = model.embed_tokens
        self.layers = model.layers
        self.norm = model.norm
        self.rotary_emb = model.rotary_emb

    def forward(self, input_ids: torch.LongTensor = None, position_ids = None, 
                past_key_values = None, inputs_embeds = None, use_cache = None, 
                output_attentions = None, output_hidden_states = None, cache_position = None ):

        # wte 
        inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache) and not self.training:
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        
        # wpe 
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)


        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # transformers
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
        
        # norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        return hidden_states

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

class LoRA_Llama_Server(nn.Module):
    def __init__(self, model:nn.Module, split_id:int):
        super().__init__()
        self.vocab_size =  model.vocab_size
        self.lm_head = model.lm_head

    def forward(self, hidden_states: torch.LongTensor = None, label_ids: torch.LongTensor = None):
        logits = self.lm_head(hidden_states).float()

        loss = None
        if label_ids is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits
        )

        

# class LoRA_Llama_Server(PreTrainedModel):
#     def __init__(self, model:nn.Module, split_id:int):
#         super().__init__(config)
#         self.padding_idx = config.pad_token_id
#         self.vocab_size = config.vocab_size

#         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
#         self.layers = nn.ModuleList(
#             [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
#         )
#         self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.rotary_emb = LlamaRotaryEmbedding(config=config)
#         self.gradient_checkpointing = False

#         # Initialize weights and apply final processing
#         self.post_init()



#     def forward(self, input_ids, position_ids=None, past=None):
#         # process past
#         if past is None:
#             past_length = 0 
#             past = [None] * len(self.split_h)
#         else:
#             past_length = past[0].size(-2)
        
#         # process position
#         if position_ids is None:
#             position_ids = torch.arange(past_length, past_length + input_ids.size(-1), 
#                                         dtype=torch.long,
#                                         device=input_ids.device)
#             position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

#         # hedden_state = text embeddings and position embeddings
#         inputs_embeds = self.wte(input_ids)
#         position_embeds = self.wpe(position_ids)
#         hidden_states = inputs_embeds + position_embeds

#         # pass data through transformer blocks
#         presents = []
#         for block, layer_past in zip(self.split_h, past):
#             hidden_states, present = block(hidden_states, layer_past)
#             presents.append(present)

#         if presents == []:
#             presents.append(position_ids.unsqueeze(-1))

#         return hidden_states, presents

# class LoRA_GPT_Server(nn.Module):
#     def __init__(self, model, split_id):
#         super(LoRA_GPT_Server, self).__init__()
#         self.split_h = nn.ModuleList()
#         if split_id<len(model.transformer.h):
#             for i in range(split_id, len(model.transformer.h)):
#                 self.split_h.add_module(str(i), model.transformer.h[i])

#         self.ln_f = model.transformer.ln_f
#         self.decoder = model.head

#     def forward(self, hidden_states, past=None, label_ids=None):
#         # process past
#         if past is None:
#             past = [None] * len(self.split_h)

#         # iterate through transformers
#         presents = []
#         try:
#             for block, layer_past in zip(self.split_h, past):
#                 hidden_states, present = block(hidden_states, layer_past)
#                 presents.append(present)
#         except:
#             pass
    
#         # pass through layer_norm + decoder
#         hidden_states = self.ln_f(hidden_states)
#         logits = self.decoder(hidden_states)

#         # calculate loss if has label, else return predict logits
#         if label_ids is not None:
#             loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
#             loss = loss_fct(logits.view(-1, logits.size(-1)), label_ids.view(-1))
#             return loss
#         return logits, presents


# class LoRA_GPT_Split_Combine(nn.Module):
#     def __init__(
#         self,
#         LoRA_GPT_Client,
#         LoRA_GPT_Server,
#     ):
#         super().__init__()
#         self.LoRA_GPT_Client = LoRA_GPT_Client
#         self.LoRA_GPT_Server = LoRA_GPT_Server

#     def forward(self, input_ids, position_ids, label_ids):
#         outputs_cl, _ = self.LoRA_GPT_Client(input_ids, position_ids)
#         loss = self.LoRA_GPT_Server(outputs_cl, label_ids=label_ids)
#         return loss



# class LlamaForCausalLM(LlamaPreTrainedModel):
#     _tied_weights_keys = ["lm_head.weight"]

#     def __init__(self, config):
#         super().__init__(config)
#         self.model = LlamaModel(config)
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()


#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
#         r"""
#         Args:
#             labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#                 config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#                 (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

#         Returns:

#         Example:

#         ```python
#         >>> from transformers import AutoTokenizer, LlamaForCausalLM

#         >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
#         >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

#         >>> prompt = "Hey, are you conscious? Can you talk to me?"
#         >>> inputs = tokenizer(prompt, return_tensors="pt")

#         >>> # Generate
#         >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
#         >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
#         ```"""
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             cache_position=cache_position,
#         )

#         hidden_states = outputs[0]
#         if self.config.pretraining_tp > 1:
#             lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
#             logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
#             logits = torch.cat(logits, dim=-1)
#         else:
#             logits = self.lm_head(hidden_states)
#         logits = logits.float()

#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#     def prepare_inputs_for_generation(