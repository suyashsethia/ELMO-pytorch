# importing the libraries 
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


# defing the model which we are going to pretrain
class ELMo(nn.Module):
    '''this class implements the ELMo model using the BI-LSTM architecture like by stacking two LSTM layers 
    the model is just the head and needs body such as linear layer , mlp , etc based on the task  '''
    def __init__(self, embedding_dim,  hidden_dim1,batch_size, embedding_matrix=None):
        super(ELMo, self).__init__()
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.embedding= nn.Embedding.from_pretrained(embedding_matrix)
        self.embedding.weight.requires_grad = False
        hidden_dim2 = embedding_dim//2
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim1, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim1*2, hidden_dim2, num_layers=1, batch_first=True, bidirectional=True)
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.lambda1 = nn.Parameter(torch.randn(1))


    def forward(self, input): 
        # input = [batch_size, seq_len]
        # getting the imput embeddings 
        input_embeddings = self.embedding(input) # [batch_size, seq_len, embedding_dim]
        # passing the embeddings to the first LSTM layer
        output1 , (hidden1, cell1) = self.lstm1(input_embeddings) # [batch_size, seq_len, hidden_dim1]

        # passing the output of the first LSTM layer to the second LSTM layer
        output2 , (hidden2, cell2) = self.lstm2(output1) # [batch_size, seq_len, hidden_dim2]
        # adding the two outputs of the LSTM layers
        
        weighted_output = self.lambda1*((self.weight1 * output1) +( self.weight2 * output2))

        return weighted_output
        


def create_elmo(embedding_matrix, batch_size,embedding_dim, hidden_dim1 ):
    '''this function creates the elmo model and returns it'''
    model = ELMo(embedding_dim=embedding_dim, hidden_dim1=hidden_dim1, batch_size=batch_size, embedding_matrix=embedding_matrix)
    return model

