
from torch.utils.data import Dataset, DataLoader
from masker_util import masking
import pandas as pd
from enum import Enum
from ast import literal_eval
from tokenizer_util import *
from sklearn.model_selection import train_test_split
from masker_util import masking, creating_targets_from_labels, TargetandMaskType
from sklearn.model_selection import train_test_split
import os

class Path(Enum):
    path_to_df = os.path.expanduser("~/Data/DataFrameforGPT2.csv")


class InfillingDataset(Dataset):
    def __init__(self, text, identities, max_len):
        super(InfillingDataset, self).__init__()
        self.text = text
        self.identities = identities
        self.max_len = max_len
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        text = self.text[idx]
        identities = self.identities[idx]
        inputs, labels = masking(text, identities)
        tokenizer = get_tokenizer()
        inputs_max_len = tokenizer.encode('<|PAD|>') * self.max_len # tokenizer.encode returns a list with the token id
        inputs_max_len[:len(inputs)] = inputs
        labels_max_len = [-1] * self.max_len
        labels_max_len[:len(labels)] = labels
        targets_context = creating_targets_from_labels([TargetandMaskType.CONTEXT],labels_max_len, inputs_max_len)
        targets_labels = creating_targets_from_labels([TargetandMaskType.CONTEXT_PHRASE, TargetandMaskType.CONTEXT_WORD, 
                                                TargetandMaskType.MASKTYPE_PHRASE, TargetandMaskType.MASKTYPE_WORD], labels_max_len, inputs_max_len)
        return torch.tensor(inputs_max_len), torch.tensor(targets_context), torch.tensor(targets_labels)
  
       
def get_data():
    df = pd.read_csv(Path.path_to_df.value)
    df['choices'] = df['choices'].apply(lambda x: literal_eval(x))
    text = df['context'].to_list()
    identities = [identity[0] for identity in list(df['choices'])]
    text_train, text_val, identities_train, identities_val = train_test_split(text, identities, test_size= 0.2 , random_state= 42, shuffle= True)
    dataset_train = InfillingDataset(text_train, identities_train, max_len = 512)
    dataloader_train = DataLoader(dataset_train, batch_size= 16, shuffle = True)
    dataset_val = InfillingDataset(text_val, identities_val, max_len = 512)
    dataloader_val = DataLoader(dataset_val, batch_size = 16, shuffle= True)
    return dataloader_train, dataloader_val

## For testing purposes
def display_data():
    dataloader_train,_ = get_data()
    for input, labels_context, labels_targets in dataloader_train:
        print(input)
        print(labels_context)
        print(labels_targets)

if __name__ == '__main__':
    display_data()








    


    




