import os
import random
from typing import List

import pandas as pd
import numpy as np

## Modelling
from transformers import AutoTokenizer

import torch
from torch.utils.data import (
    TensorDataset,
    RandomSampler,
    SequentialSampler,
    DataLoader
)

## Metrics / Utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

def get_length(df: pd.DataFrame, texts_col: int):
    '''
    Max sequence length for training a NN

    Based on heuristic (mean and std over the length distribution of texts)
    '''
    df[texts_col] = df[texts_col].astype(str)

    sequences_length = df[texts_col].apply(lambda x: len(x.split()))

    max_seq_length = int(round(sequences_length.mean() + sequences_length.std()))

    return max_seq_length

def get_labels(df: pd.DataFrame, labels_col):
    '''
    Encode labels from df

    return np.array of labels
    '''
    LB = LabelEncoder()

    LB.fit(df[labels_col])

    return LB.transform(df[labels_col])

def encode_texts(df: pd.DataFrame, texts_col: str, tokenizer: str = "bert-base-uncased", max_seq_length: int = 512, return_vocab_size: bool = True):
    """"
    Encode list of texts using pretrained tokenizer from huggingface

    return np.array of encoded sequence 
    """
    pretrained_tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)

    texts = list(df[texts_col].astype(str))

    encoded_sequence = pretrained_tokenizer.batch_encode_plus(texts, 
                                                              add_special_tokens=True, 
                                                              pad_to_max_length=True, 
                                                              max_length=max_seq_length,
                                                              return_attention_masks=False,
                                                              return_token_type_ids=False)['input_ids']
    return encoded_sequence, pretrained_tokenizer.vocab_size

def create_TorchLoaders(X: List = None, y: np.array = None, test_size: int = 0.10, batch_size: int = 32, batch_size_eval: int = 64):
    '''

    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.float)
    )

    train_sampler = RandomSampler(train_dataset)

    test_sampler = SequentialSampler(test_dataset)

    train_loader = DataLoader(
        dataset = train_dataset,
        sampler = train_sampler,
        batch_size =batch_size
        )

    test_loader = DataLoader(
        dataset = test_dataset,
        sampler = test_sampler,
        batch_size = batch_size_eval
    )

    return train_loader, test_loader
