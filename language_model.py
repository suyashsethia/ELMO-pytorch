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

class Language_model(nn.Module):
    '''this class implements the language model using the ELMo model as the head and a linear layer as the body'''
    def __init__(self, Elmo_model, vocab_size, embedding_dim):
        super(Language_model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.elmo = Elmo_model
        self.linear = nn.Linear(self.embedding_dim, self.vocab_size)
    def forward(self, input):
        # input = [batch_size, seq_len]
        # getting the imput embeddings 
        elmo_output = self.elmo(input) # [batch_size, seq_len, embedding_dim]
        output = self.linear(elmo_output) # [batch_size, seq_len, vocab_size]
        output = F.log_softmax(output, dim=2).permute(0,2,1)[:,:,:-1] # [batch_size, vocab_size, seq_len-1]
        return output
    


def train (MODEL , pretrain_loaders, LEARNING_RATE, DEVICE , EPOCHS):
    '''pretrain loaders is a dictionary containing the train and validation and test loaders for the pretraining task
    this function trains the MODEL on the pretraining task
    '''
    MODEL.to(DEVICE)
    criterion = nn.NLLLoss()

    # define the optimizer 
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
    best_loss = 1000000
    best_accuracy = 0
    def accuracy(output, label):
        output = output.argmax(dim=1)
        return (output == label).float().mean()
    steps = 0
    running_loss = 0
    for epoch in range(EPOCHS):
        print('epoch: ', epoch)
        if epoch%3 == 0 and epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/2
        for input, label in pretrain_loaders['train']:
            steps += 1
            optimizer.zero_grad()
            MODEL.zero_grad()
            input = input.to(DEVICE)
            label = label.to(DEVICE)
            output = MODEL.forward(input)
            loss = criterion(output, label)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if steps%15 == 0:
                MODEL.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_accuracy = 0
                    for input, label in pretrain_loaders['validation']:
                        input = input.to(DEVICE)
                        label = label.to(DEVICE)
                        output = MODEL.forward(input)
                        val_loss += criterion(output, label)
                        val_accuracy += accuracy(output, label)
                    val_loss = val_loss/len(pretrain_loaders['validation'])
                    val_accuracy = val_accuracy/len(pretrain_loaders['validation'])
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(MODEL.state_dict(), 'best_loss.pth')
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        torch.save(MODEL.state_dict(), 'best_accuracy.pth')
                    print( 'train loss: ', running_loss/100, 'validation loss: ', val_loss, 'validation accuracy: ', val_accuracy)
                    running_loss = 0
                MODEL.train()

def make_language_model(Elmo_model, vocab_size, embedding_dim):
    '''this function makes the language model and trains it on the pretraining task'''
    MODEL = Language_model(Elmo_model, vocab_size, embedding_dim)
    return MODEL