from pathlib import Path
import torch
from official_gpt2_encoder.encoder import Encoder, get_encoder
import json
import os
from enum import Enum
from transformers import GPT2Tokenizer

class Path(Enum):
    #Insert the path here
    PATH = os.path.expanduser('E:/ResearchWork/CurrentResearch/gpt2-ilm/InfillGPT2/official_gpt2_encoder')
    path_to_vocab_file = os.path.join(PATH, 'encoder.json')
    path_to_merges_file = os.path.join(PATH, "vocab.bpe")


additional_tokens = {
    'start_of_infill_token': '<|startofinfill|>',
    'infill_word' : '<|infillword|>',
    'infill_phrase': '<|infillphrase|>',
    'end_of_infill_token': '<|endofinfill|>',
    'pad_token' : '<|PAD|>'
}
# This function is to ensure that the tokenizer is returned
def get_tokenizer():
    #print(Path.path_to_vocab_file.value)
    tokenizer = GPT2Tokenizer(vocab_file= Path.path_to_vocab_file.value, merges_file = Path.path_to_merges_file.value)
    update_tokenizer(tokenizer) # Call update and then initialize the tokenizer 
    return tokenizer
    

#Update the vocab file

def update_tokenizer(tokenizer):
    list_of_tokens = [v for _,v in additional_tokens.items()]
    tokenizer.add_tokens(list_of_tokens)

def tokenize(text):
    tokenizer = get_tokenizer()
    #update_tokenizer(tokenizer)
    input_ids = tokenizer(text)['input_ids']
    # This tokenized text will be used while masking
    tokenized_text = [tokenizer.decode(x) for x in input_ids]
    return input_ids, tokenized_text, tokenizer

if __name__ == '__main__':
    tokenizer = get_tokenizer()
    print(tokenizer.encode("<|PAD|>"))
















