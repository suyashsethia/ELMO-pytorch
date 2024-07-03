import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from datasets import load_dataset
import numpy as np
import pandas as pd
import random
from torch import cuda
from pprint import pprint
import re 


class PretrainDataset(Dataset):
    '''This class helps in creating the Pretrain Dataset'''
    def __init__(self, data):
        random.shuffle(data)
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx]['sentence']),torch.tensor( self.data[idx]['label']))
    

class NLIDataset(Dataset):
    '''This is for creating the NLI Dataset'''
    def __init__(self, data):
        random.shuffle(data)
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]['sentence'][0]),torch.tensor(self.data[idx]['sentence'][1]) ,torch.tensor( self.data[idx]['label'])



def custom_collate_sman(batch):
    '''Custom collate function for semanitc analysis task'''
    sentences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    for i , label in enumerate(labels):
        if label <=0.5:
            labels[i]=0
        else :
            labels[i]=1

    # Pad sequences to the maximum length in the batch
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)
    labels = torch.LongTensor(labels)
    
    return  padded_sentences,labels



def custom_collate(batch):
    '''Custom collate function for pretraining task'''
    sentences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Pad sequences to the maximum length in the batch
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return ( padded_sentences,padded_labels)



def custom_collate_nli(batch):
    ''' Custom collate function for Natural Language Inference '''
    premises, hypothesis = [item[0] for item in batch], [item[1] for item in batch]
    labels = [item[2] for item in batch]
  
    # Pad sequences to the maximum length in the batch
    padded_premises = torch.nn.utils.rnn.pad_sequence(premises, batch_first=True)
    padded_hypothesis = torch.nn.utils.rnn.pad_sequence(hypothesis, batch_first=True)
    labels = torch.LongTensor(labels)
    
    return (torch.tensor(padded_premises), torch.tensor(padded_hypothesis)),labels



def preprocessing(sentence, stop_words_remove, word_2_idx, stop_words):
    '''This function helps in preprocessing the sentence
    Args:
        sentence: Sentence to be preprocessed  
        stop_words_remove: Boolean value to remove stop words or not
    Returns:
        sentence: Preprocessed sentence in form of list of tokens with idx values'''
    sentence = sentence.split(' ')
    if stop_words_remove:
        sentence = [word.lower() for word in sentence if word.lower() not in stop_words]
    else:
        sentence = [word.lower() for word in sentence]
    sentence = ['<start> '] + sentence+ ['<end>']
    sentence = [word_2_idx[word] if word in word_2_idx else word_2_idx['<unk>'] for word in sentence]
    return sentence








