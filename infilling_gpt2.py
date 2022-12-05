from numpy import require
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.optimization import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tokenizer_util import get_tokenizer
from dataset_loading import get_data
from enum import Enum
import argparse
import os

class ModelPath(Enum):
    PATH = os.path.expanduser("~/ModelState") # This parameter will be set in AWS

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_model():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = get_tokenizer()
    model.resize_token_embeddings(len(tokenizer))
    return model

def train(epochs):
    model = get_model()
    model.load_state_dict(torch.load(os.path.join(ModelPath.PATH.value, 'GPT2FineTuned.pt'))) 
    dataloader_train, dataloader_val = get_data()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr = 1e-03)
    print(device)
    model.train()
    total_loss_context, total_loss_infill = 0, 0
    for epoch in range(1, epochs + 1):
        try:
            model.train()
            for i, data in enumerate(dataloader_train):
                #batch_data = data.to(device)
                inputs, labels_context, labels_infill = data
                #print(inputs)
                inputs = inputs.to(device)
                labels_context = labels_context.to(device)
                labels_infill = labels_infill.to(device)
                loss_context = model(inputs, labels = labels_context)[0]
                loss_infill = model(inputs, labels = labels_infill)[0]
                print("Losses for batch %0.0f are %0.4f and %0.4f" %(i+1, loss_context, loss_infill))
                optimizer.zero_grad()
                total_loss_context += loss_context.item()
                total_loss_infill += loss_infill.item()
                loss_context.backward(retain_graph = True)
                loss_infill.backward()
               # nn.utils.clip_grad_norm(model.parameters(), max_norm = 2.0, norm_type = 2)
                optimizer.step()
                inputs = None
                labels_infill = None
                labels_context = None
            
            
            #if epoch % 5 == 0:
            total_loss_context = total_loss_context / len(dataloader_train)
            total_loss_infill = total_loss_infill / len(dataloader_train)
            print('Context Loss  and Infill Loss at epoch %d are %0.4f, %0.4f' %(epoch, total_loss_context, total_loss_infill))
            eval(model, dataloader_val)
            if epoch == epochs:
            # Save the model
                torch.save(model.state_dict(), os.path.join(ModelPath.PATH.value, 'GPT2FineTuned.pt'))
        except KeyboardInterrupt:
            torch.save(model.state_dict(), os.path.join(ModelPath.PATH.value, 'GPT2FineTuned.pt'))

    



def eval(model, dataloader_val):
    model.eval()
    total_val_loss_context, total_val_loss_infill = 0, 0

    with torch.no_grad():
        for i, data in enumerate(dataloader_val):
            inputs, labels_context, labels_infill = data
            inputs = inputs.to(device)
            labels_context = labels_context.to(device)
            labels_infill = labels_infill.to(device)
            loss_context = model(inputs,labels =  labels_context)[0]
            loss_infill = model(inputs, labels =  labels_infill)[0]
            #labels_context.detach()
            #labels_infill.detach()
            total_val_loss_context += loss_context.item()
            total_val_loss_infill += loss_infill.item()
            inputs = None
            labels_context = None
            labels_infill = None
            print("Validation context loss and validation infill loss is %0.4f and %0.4f for batch %0.0f" %(loss_context, loss_infill, i+1))
        
    print('Total validation  context loss and total validation infill loss is %0.4f and %0.4f' %(total_val_loss_context / len(dataloader_val) , total_val_loss_infill/len(dataloader_val)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type= int, required = True)
    args = parser.parse_args()
    epochs = args.epochs
    train(epochs)





        


    




    


