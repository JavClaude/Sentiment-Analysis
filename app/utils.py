import os
import sys

sys.path.insert(0, "../src/")

import json
import torch
from transformers import AutoTokenizer

from model import LSTMModel

def load_model(path_to_state_dict: str, path_to_config_file: str):

    with open(path_to_config_file, 'r') as file:
        config_file = json.load(file)
    
    Model = LSTMModel(**config_file)
    Model.load_state_dict(torch.load(path_to_state_dict))

    Model.eval() #LN in eval mode

    return Model, config_file

def load_tokenizer(tokenizer: str = "bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    return tokenizer
    
def encode_text(text, tokenizer, config_file):
    assert type(text) == str, "Please provide a string"
    texts_to_ids = tokenizer.encode_plus(text,
                                         add_special_tokens=True,
                                         pad_to_max_length=True,
                                         max_length=config_file['max_length'],
                                         return_attention_mask=False,
                                         return_token_type_ids=False)['input_ids']

    return torch.tensor(texts_to_ids, dtype=torch.long)
    
def predict(model, tensor):
    Sigmoid = torch.nn.Sigmoid()
    preds = Sigmoid(model(tensor.view(1, -1)))
    return preds.detach().numpy()
