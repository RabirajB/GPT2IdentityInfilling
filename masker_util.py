from enum import Enum
from tracemalloc import start
import nltk
from numpy import dtype
#from zmq import context
nltk.download('stopwords')
from nltk.corpus import stopwords
from tokenizer_util import tokenize, get_tokenizer
import torch
from ast import literal_eval
import pandas as pd


stopwords = stopwords.words('english')

class TargetandMaskType(Enum):
    #Placeholders
    CONTEXT = 1
    MASKTYPE_WORD = 2
    MASKTYPE_PHRASE = 3
    CONTEXT_WORD = 4
    CONTEXT_PHRASE = 5
    START_of_INFILL = 6
    END_OF_INFILL = 7


def get_positions_for_identity(identity, text):
    input_ids, tokenized_text, _ = tokenize(text)
    positions = []
    #print(tokenized_text)
    #print(input_ids)
    count_match = []
    for i in range(len(tokenized_text)):
        if tokenized_text[i].strip() == identity:
            positions.append(i)
            break

        elif tokenized_text[i].strip() in identity and tokenized_text[i].strip() not in stopwords and len(tokenized_text[i].strip()) > 1:
           # print(tokenized_text[i])
            if identity.index(tokenized_text[i].strip()) not in count_match:
                positions.append(i)
                count_match.append(identity.index(tokenized_text[i].strip()))
            else:
                continue

    return positions

def masking(text, identity):
    positions = get_positions_for_identity(identity, text)
    input_ids, tokenized_text, tokenizer = tokenize(text)
    #print(input_ids)
    inputs, labels = [], []
    #print(inputs)
    identity_split = identity.split()
    #target_infill_word_id, target_infill_word, target_infill_phrase_id, target_infill_phrase = 0, 0, 0, 0
    target_infill_id, target_type, context_type = 0, 0, 0
    start_of_infill_id, end_of_infill_id = tokenizer.encode('<|startofinfill|>')[0], tokenizer.encode('<|endofinfill|>')[0]
    #print(identity_split)
    if len(identity_split) > 1:
        target_infill_id = tokenizer.encode('<|infillphrase|>')[0] # Get the id from a single element list
        target_type = TargetandMaskType.MASKTYPE_PHRASE.value
        context_type = TargetandMaskType.CONTEXT_PHRASE.value
    else:
        target_infill_id = tokenizer.encode('<|infillword|>')[0] # Get the id from the single element list
        target_type = TargetandMaskType.MASKTYPE_WORD.value
        context_type = TargetandMaskType.CONTEXT_WORD.value
    count = 0
    for i in range(len(tokenized_text)):
        if i not in positions:
            inputs.append(input_ids[i])
            labels.append(TargetandMaskType.CONTEXT.value)
        else:
            if target_type == TargetandMaskType.MASKTYPE_PHRASE.value and count != 1:
                inputs.append(target_infill_id)
                labels.append(target_type)
                count += 1
            elif target_type == TargetandMaskType.MASKTYPE_WORD.value and count != 1:
                inputs.append(target_infill_id)
                labels.append(target_type)
                count += 1
            else:
                continue
            
    count_pos = 0
    if context_type == TargetandMaskType.CONTEXT_PHRASE.value:
        for pos in positions:
            if count_pos == 0:
                inputs.extend([start_of_infill_id, input_ids[pos]])
                labels.extend([TargetandMaskType.START_of_INFILL.value, TargetandMaskType.CONTEXT_PHRASE.value])
                count_pos += 1
            elif count_pos < len(positions) - 1:
                inputs.append(input_ids[pos])
                labels.append(TargetandMaskType.CONTEXT_PHRASE.value)
                count_pos += 1
            elif count_pos == len(positions) - 1:
                inputs.extend([input_ids[pos], end_of_infill_id])
                labels.extend([TargetandMaskType.CONTEXT_PHRASE.value,TargetandMaskType.END_OF_INFILL.value])
    if context_type == TargetandMaskType.CONTEXT_WORD.value:
        for pos in positions:
            if count_pos == 0:
                inputs.extend([start_of_infill_id, input_ids[pos]])
                labels.extend([TargetandMaskType.START_of_INFILL.value, TargetandMaskType.CONTEXT_WORD.value])
                count_pos += 1
            elif count_pos < len(positions) - 1:
                #print('2nd If Triggered')
                inputs.append(input_ids[pos])
                labels.append(TargetandMaskType.CONTEXT_WORD.value)
                count_pos += 1
            elif count_pos == len(positions) - 1:
                inputs.extend([input_ids[pos], end_of_infill_id])
                labels.extend([TargetandMaskType.CONTEXT_WORD.value,TargetandMaskType.END_OF_INFILL.value])         

    return inputs, labels


def creating_targets_from_labels(target_types, labels, inputs):
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    #print(inputs)
    selection = torch.zeros_like(inputs , dtype = torch.bool)
    for target_type in target_types:
        selection = selection | (labels == target_type.value)
    #print(selection)
    return torch.where(selection,
                        inputs,
                        torch.full_like(inputs, -1)
                       )


if __name__ == '__main__':

    text, identities = "Rhodri Williams is a Welsh sports journalist from Barry, Vale of Glamorgan, Wales.", 'Welsh sports journalist'
    positions = get_positions_for_identity(identities, text)
    print(positions)
    inputs, labels = masking(text, identities)
    print(inputs)
    print(labels)
    labels_context = creating_targets_from_labels([TargetandMaskType.CONTEXT], labels, inputs)
    print(labels_context)   
    labels_infill = creating_targets_from_labels([TargetandMaskType.CONTEXT_PHRASE, TargetandMaskType.CONTEXT_WORD,
                                                     TargetandMaskType.END_OF_INFILL], labels, inputs)
    print(labels_infill)









    


    
