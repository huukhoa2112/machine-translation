# Import library
import pandas as pd
import numpy as np 
import re
import random
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.lang.vi import Vietnamese
from iteration_utilities import deepflatten
from nltk.translate.bleu_score import sentence_bleu
from torch import nn
import torch
import torchtext
from torchtext.legacy import data
# from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator, Dataset
from torch.nn  import functional as F
import torch.optim as  optim 

if torch.cuda.is_available():  
    dev = "cuda:0" 

    print("gpu up")
else:  
    dev = "cpu"  
device = torch.device(dev)
spacy_en = spacy.load("en_core_web_sm")
SEED = 32

enNLP = English()
viNLP = Vietnamese()

enTokenizer = Tokenizer(enNLP.vocab)
viTokenizer =  Tokenizer(viNLP.vocab)

def create_raw_dataset():
    data_dir = "data/"
    en_sents = open(data_dir + 'train_en.txt', "r").read().splitlines()
    vi_sents = open(data_dir + 'train_vi.txt', "r").read().splitlines()
    return {
        "english": [line for line in en_sents],
        "vietnamese": [line for line in vi_sents],
    }

def myTokenizerEN(x):
    return  [word.text for word in 
          enTokenizer(re.sub(r"\s+\s+"," ",re.sub(r"[\.\'\`\"\r+\n+]"," ",x.lower())).strip())]
def myTokenizerVI(x):
    return  [word.text for word in 
          viTokenizer(re.sub(r"\s+\s+"," ",re.sub(r"[\.\'\`\"\r+\n+]"," ",x.lower())).strip())]

class DataFrameDataset(data.Dataset):

    def __init__(self, df, src_field, target_field, is_test=False, **kwargs):
        fields = [('english', src_field), ('vietnamese',target_field)]
        examples = []
        for i, row in df.iterrows():
            eng = row.english 
            ar = row.vietnamese
            examples.append(data.Example.fromlist([eng, ar], fields))

        super().__init__(examples, fields, **kwargs)

class TranslateTransformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        max_len,
    ):
        super(TranslateTransformer, self).__init__()
        self.srcEmbeddings = nn.Embedding(src_vocab_size,embedding_size)
        self.trgEmbeddings= nn.Embedding(trg_vocab_size,embedding_size)
        self.srcPositionalEmbeddings= nn.Embedding(max_len,embedding_size)
        self.trgPositionalEmbeddings= nn.Embedding(max_len,embedding_size)
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.src_pad_idx = src_pad_idx
        self.max_len = max_len
    
    def make_src_mask(self, src):
        src_mask = src.transpose(0,1) == self.src_pad_idx

        return src_mask.to(device)

    def forward(self,x,trg):
        src_seq_length = x.shape[0]
        N = x.shape[1]
        trg_seq_length = trg.shape[0]
        #adding zeros is an easy way
        src_positions = (
            torch.arange(0, src_seq_length)
            .reshape(src_seq_length,1)  + torch.zeros(src_seq_length,N) 
        ).to(device)
        
        trg_positions = (
            torch.arange(0, trg_seq_length)
            .reshape(trg_seq_length,1)  + torch.zeros(trg_seq_length,N) 
        ).to(device)


        srcWords = self.dropout(self.srcEmbeddings(x.long()) +self.srcPositionalEmbeddings(src_positions.long()))
        trgWords = self.dropout(self.trgEmbeddings(trg.long())+self.trgPositionalEmbeddings(trg_positions.long()))
        
        src_padding_mask = self.make_src_mask(x)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(device)
        
        
        out = self.transformer(srcWords,trgWords, src_key_padding_mask=src_padding_mask,tgt_mask=trg_mask)
        out= self.fc_out(out)
        return out
    
def translate(model,sentence,srcField,targetField,srcTokenizer):
    model.eval()
    processed_sentence = srcField.process([srcTokenizer(sentence)]).to(device)
    trg = ["<sos>"]
    for _ in range(60):
        
        trg_indecies = [targetField.vocab.stoi[word] for word in trg]
        outputs = torch.Tensor(trg_indecies).unsqueeze(1).to(device)
        outputs = model(processed_sentence,outputs)
        
        if targetField.vocab.itos[outputs.argmax(2)[-1:].item()] == "<unk>":
            continue 
        trg.append(targetField.vocab.itos[outputs.argmax(2)[-1:].item()])
        if targetField.vocab.itos[outputs.argmax(2)[-1:].item()] == "<eos>":
            break
    return " ".join([word for word in trg if word != "<unk>"][1:-1])

# def run_once(f):
#     def wrapper(*args, **kwargs):
#         if not wrapper.has_run:
#             wrapper.has_run = True
#             return f(*args, **kwargs)
#     wrapper.has_run = False
#     return wrapper

# @run_once
# def build():

#     raw_data = create_raw_dataset()
#     df = pd.DataFrame(raw_data)

#     SRC = data.Field(tokenize=myTokenizerEN,batch_first=False,init_token="<sos>",eos_token="<eos>")
#     TARGET = data.Field(tokenize=myTokenizerVI,batch_first=False,tokenizer_language="vi",init_token="<sos>",eos_token="<eos>")        
#     torchdataset = DataFrameDataset(df,SRC,TARGET)

#     train_data, valid_data = torchdataset.split(split_ratio=0.8, random_state = random.seed(SEED))
#     SRC.build_vocab(train_data,min_freq=2)
#     TARGET.build_vocab(train_data,min_freq=2)  
#     return SRC, TARGET
    

import streamlit as st

# def train_en_vi(SRC, TARGET):

src = torch.load('source.pt')
tar = torch.load('target.pt')
model = torch.load('translate_en_vi.pt')
eng_sentence = st.text_input(label='Your English sentence:', placeholder='Type here')
vi_sentence = translate(model,eng_sentence ,src,tar,myTokenizerEN)
st.write('Vietnamese sentence: ', vi_sentence)