import os
import random
from typing import List

import pandas as pd

## Modelling
from transformers import AutoTokenizer

## Metrics / Utils
from sklearn.preprocessing import LabelEncoder

def get_length(df: pd.DataFrame, col_index: int):
    '''
    Max sequence length for training a NN

    Based on heuristic (mean and std over the length distribution of texts)
    '''
    df[df.columns[col_index]] = df[df.columns[col_index]].astype(str)

    sequences_length = df[df.columns[col_index]].apply(lambda x: len(x.split()))

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

def encode_texts(df: pd.DataFrame, texts_col: str, tokenizer: str = "bert-base-uncased", max_seq_length: int = 512):
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
    return encoded_sequence
