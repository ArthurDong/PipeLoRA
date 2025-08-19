#####################################################################################################
#                                      Load Data as DataLoader                                      #
#####################################################################################################
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai--deepseek-llm-7b-base")
tokenizer.pad_token = tokenizer.eos_token

class LLAMATextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=4096):
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


# datasets
from datasets import load_dataset
def get_dataloader():
    dataset = load_dataset("text", data_files={"train": "train.txt", "valid": "val.txt"})
    train_datasets = LLAMATextDataset(dataset["train"]["text"], tokenizer, 128)
    valid_datasets = LLAMATextDataset(dataset["valid"]["text"], tokenizer, 128)
    return {"train":DataLoader(train_datasets, batch_size=1, shuffle=True),
            "valid":DataLoader(valid_datasets, batch_size=1)}

# get dataloader for training
def get_loader(filename='./datasets/all_the_news/articles1.csv', batch_size:int=100, n_records:int=50000, valid_ratio:float=0.2, max_length=1024, train_sampler=None):
    df = pd.read_csv(filename)[0:int(n_records)]
    texts = df['content'].dropna().tolist()

    train_text = texts[0:int((1-valid_ratio) * n_records)]
    valid_text = texts[int(valid_ratio * n_records):]

    train_dataset = LLAMATextDataset(train_text, tokenizer, max_length=max_length)
    valid_dataset = LLAMATextDataset(valid_text, tokenizer, max_length=max_length)

    return {"train":DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=train_sampler),
            "valid":DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)}


from torch.utils.data import RandomSampler
class RandomSamplerWithOneAddtionalIteration(RandomSampler):
    def  __init__(self, data_source, replacement = False, num_samples = None, generator=None):
        super().__init__(data_source, replacement, num_samples, generator)

    def __iter__(self):
        data_source = list(super().__iter__())
        data_source.append(self.__len__())
        return iter(data_source)

    def __len__(self) -> int:
        return super().__len__() + 1

# df = pd.read_csv('./datasets/all_the_news/articles1.csv')[0:40]
# texts = df['content'].dropna().tolist()
# dataset = LLAMATextDataset(texts, tokenizer, max_length=128)
# torch.manual_seed(42)
# print(list(RandomSampler(dataset)))
# torch.manual_seed(42)
# print(list(RandomSampler(dataset)))
# torch.manual_seed(42)
# print(list(RandomSamplerWithAddtionalIteration(dataset)))

#####################################################################################################
#                                         pred_text_llama2                                          #
#####################################################################################################

def pred_text_deepseek(prompt_text:str, model:torch.nn.Module, max_length:int=50):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai--deepseek-llm-7b-base")
    tokenizer.pad_token = tokenizer.eos_token

    # Text to Token
    text_token = tokenizer(prompt_text)
    input_ids = torch.tensor(text_token['input_ids'], dtype=torch.long).unsqueeze(0).to("cpu")

    if isinstance(model, torch.nn.Module):
        model.eval()
    else:
        model[0].eval()
        model[1].eval()

    for _ in range(max_length):

        if isinstance(model, torch.nn.Module):
            pred_token = model(input_ids = input_ids.to(model.device))
        else:
            hidden_states = model[0](input_ids = input_ids.to(model[0].device))["hidden_states"]
            pred_token = model[1](hidden_states = hidden_states.to(model[1].device), label_ids=input_ids)

        last_token_logits = pred_token["logits"][:, -1, :]
        next_token_id = torch.argmax(last_token_logits, dim=-1).to("cpu")
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

        # Early break if come-cross <EOS> or <\n>. "eos" id is 2; "\n" id is 13
        if next_token_id.item() == tokenizer.eos_token_id or next_token_id.item() == 13:
            break

    pred_tokens = input_ids[0]
    predict_text = tokenizer.decode(pred_tokens.squeeze(), clean_up_tokenization_spaces=True, skip_special_tokens=True)
    return predict_text.replace(tokenizer.eos_token, '')