#####################################################################################################
#                                      Load Data as DataLoader                                      #
#####################################################################################################
import torch
import pandas as pd
import random
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader, Subset

class GPT2TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.texts = [text + tokenizer.eos_token for text in texts]
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx], 
                                truncation=True, 
                                max_length=self.max_length, 
                                padding='max_length', 
                                return_tensors='pt')
        return tokens['input_ids'].squeeze(), torch.tensor(range(self.max_length))

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', clean_up_tokenization_spaces=True)
tokenizer.pad_token = tokenizer.eos_token


def get_subset(dataloader, num_samples=50):
    # Get the dataset from the DataLoader
    dataset = dataloader.dataset
    
    # Ensure that num_samples does not exceed the available data size
    total_samples = len(dataset)
    if num_samples > total_samples:
        num_samples = total_samples
    
    # Select random indices
    random_indices = random.sample(range(total_samples), num_samples)
    
    # Create a subset of the dataset
    subset = Subset(dataset, random_indices)
    
    # Create a new DataLoader for the subset
    subset_dataloader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=False)
    
    return subset_dataloader

# get dataloader for training
def get_loader(filename='./datasets/all_the_news/articles1.csv', batch_size:int=100, n_records:int=50000, valid_ratio:float=0.2, max_length=1024):
    df = pd.read_csv(filename)[0:int(n_records)]
    texts = df['content'].dropna().tolist()

    train_text = texts[0:int((1-valid_ratio) * n_records)]
    test_text = texts[0:int(valid_ratio * n_records)]

    train_dataset = GPT2TextDataset(train_text, tokenizer, max_length=max_length)
    test_dataset = GPT2TextDataset(test_text, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# encode text prompt
def text_encode(text:str="How are you."):
    text_token = tokenizer(text)
    text, attention_mask = text_token['input_ids'], text_token['attention_mask']
    return torch.tensor(text, dtype=torch.long).squeeze(), torch.tensor(attention_mask).squeeze()

# decode text token
def text_decode(token:list):
    decoded_text = tokenizer.decode(token.squeeze(), clean_up_tokenization_spaces=True)
    decoded_text = decoded_text.replace(tokenizer.eos_token, '')

    return decoded_text

# load model weight from pretrained model
def get_weight(model:torch.nn.Module, state_dict_filename='./datasets/gpt2_pretrained.pth', device="cuda"):
    state_dict = torch.load(state_dict_filename, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    return model

def pred_text(pred_text, models, device_id='cuda:0', temperature=0.7, max_length=64, top_k=50):
    models['model_cl'].eval()
    models['model_sv'].eval()

    input_ids, _ = text_encode(pred_text)
    input_ids = input_ids.unsqueeze(0).to(device_id)
    models['model_cl'].to(device_id)
    models['model_sv'].to(device_id)

    pred_ids = input_ids
    past_cl, past_sv = None, None
    for _ in range(max_length):

        output_ids, past_cl = models['model_cl'](input_ids = input_ids, past = past_cl)
        logits, past_sv = models['model_sv'](hidden_states = output_ids, past = past_sv)
        last_token_logits = logits[:, -1, :] / temperature

        # select topk values as predictions
        indices_to_remove = last_token_logits < torch.topk(last_token_logits, top_k)[0][:,-1]
        last_token_logits[indices_to_remove] = -1e10
    
        # get next text token
        probs = torch.nn.functional.softmax(last_token_logits, dim=-1) 
        input_ids = torch.multinomial(probs, num_samples=1)

        # Stop if <EOS> token is generated
        if input_ids.item() == tokenizer.eos_token_id or input_ids.item() == tokenizer.encode("\n")[0]:
            pred_ids = torch.cat([pred_ids, input_ids], dim=-1)
            break

        pred_ids = torch.cat([pred_ids, input_ids], dim=-1)

    print(f"Text Predicted: {text_decode(pred_ids)}")


def pred_text_llama(pred_text, models, temperature=0.7, max_length=64, top_k=50):
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

    models['model_cl'].eval()
    models['model_sv'].eval()

    input_ids, _ = text_encode(pred_text)
    input_ids = input_ids.unsqueeze(0)

    pred_ids = input_ids.to("cpu")
    for _ in range(max_length):

        hidden_states = models['model_cl'](input_ids = input_ids.to("cuda:0"))
        outputs = models['model_sv'](hidden_states = hidden_states.to("cuda:1"))
        last_token_logits = outputs.logits[:, -1, :] / temperature

        # next_token_id = torch.argmax(last_token_logits, dim=-1) 
        # input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

        # select topk values as predictions
        indices_to_remove = last_token_logits < torch.topk(last_token_logits, top_k)[0][:,-1]
        last_token_logits[indices_to_remove] = -1e10
    
        # get next text token
        probs = torch.nn.functional.softmax(last_token_logits, dim=-1) 
        input_ids = torch.multinomial(probs, num_samples=1).to("cpu")

        # Stop if <EOS> token is generated
        if input_ids.item() == tokenizer.eos_token_id or input_ids.item() == tokenizer.encode("\n")[0]:
            pred_ids = torch.cat([pred_ids, input_ids], dim=-1)
            break

        pred_ids = torch.cat([pred_ids, input_ids], dim=-1)

    print(f"Text Predicted: {text_decode(pred_ids)}")

#####################################################################################################
#                                     Load Model Configurature                                      #
#####################################################################################################
import torch
import json

class GPT2Config:
    def __init__(self, config=None):
        for key, value in config.items():
            setattr(self, key, value)

# get model cofig
def get_config(config_filename='./datasets/gpt2_config.json'):
    with open(config_filename) as f:
        config = json.load(f)
    return GPT2Config(config)

# load model weight from pretrained model
def get_weight(model:torch.nn.Module, state_dict_filename='./datasets/gpt2_pretrained.pth', device="cuda"):
    state_dict = torch.load(state_dict_filename, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    return model


from transformers.configuration_utils import PretrainedConfig

class LlamaConfig(PretrainedConfig):
    def __init__(self, config_filename='./datasets/llama2_config.json',**kwargs):
        with open(config_filename) as f:
            config = json.load(f)
    
        for key, value in config.items():
            setattr(self, key, value)

        super().__init__(
            **kwargs,
        )