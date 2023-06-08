import os 
import sys 
import torch
from torch.utils.data import Dataset, DataLoader
from DataLoaders      import  data_loader
import sklearn
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import torch.nn as nn



def run_epoch(model,loader,test_loader,Real_batch ,optimizer,criterion,epoch:int=1,print_every:int=1):
    loss_list       = np.array([])
    loss_list_epoch = np.array([])

    model.train()
    for Z,Y  in  loader:              # For each batch of proteins
        species = Y.shape[1]                # Get the number of proteins in the batch
        NB  = species//Real_batch   - 1     # Calculate number of batches
        for i in range(0,len(Z)):           # For each protein in the batch
            Z_i   = Z[:,i,:]                # Get the ith protein embedding
            Y_i   = Y[:,i,:]                # Get the ith protein interaction vector 
            running_loss  = 0               # Initialize loss vector
            for j in range(0,NB):           # For each batch of proteins
                optimizer.zero_grad()                                            # Clear gradients
                lim_inf  = j*Real_batch                                          # Calculate lower limit
                lim_sup  = (j+1)*Real_batch                                      # Calculate upper limit
                Z_repeat = Z_i.repeat(Real_batch,1).unsqueeze(0).permute(1,0,2)  # Repeat ith protein embedding Real_batch times
                Zj       = Z[:,lim_inf:lim_sup,:].permute(1,0,2)                 # Get   the jth batch of protein embeddings
                Xb       = torch.cat((Z_repeat ,Zj ),axis=1)                     # Concatenate the two tensors
                yb       =  Y_i[:,lim_inf:lim_sup].squeeze()                     # Get the jth batch of labels

                y_hat  = model(Xb)                                              # Forward pass
                loss   = criterion(y_hat , yb )                                 # Calculate loss
                running_loss += loss.item()                                     # Add loss to running loss
                
                loss.backward()                                                 # Backward pass
                optimizer.step()                                                # Update parameters
            loss_list = np.append(loss_list,running_loss/NB)                    # Add average loss to loss list
        loss_list_epoch = np.append(loss_list_epoch,np.mean(loss_list))         # Add average loss to loss list

        if epoch % print_every == 0:
            accuracy = evaluate_performance(model,test_loader,threshold=0.5,Real_batch=Real_batch)
            print(f"Epoch {epoch} Loss: {np.mean(loss_list)}   Accuracy: {accuracy}")
        
    ### save model state dict
    save_model_path = os.path.join('C:/Users/golde/Documents/bert2sage/saved_models',f"model_{epoch}.pt")
    torch.save(model.state_dict(), save_model_path)

    return loss_list_epoch


def evaluate_performance(model,loader,activation_function = nn.Sigmoid() ,threshold:float=0.5,Real_batch:int=100):
        model.eval()
        with torch.no_grad():
            accuracy_list       = np.array([])
            accuracy_list_epoch = np.array([])
            for Z,Y  in  loader:              # For each batch of proteins
                species = Y.shape[1]                # Get the number of proteins in the batch
                NB  = species//Real_batch   - 1     # Calculate number of batches
                for i in range(0,len(Z)):           # For each protein in the batch
                    Z_i   = Z[:,i,:]                # Get the ith protein embedding
                    Y_i   = Y[:,i,:]                # Get the ith protein interaction vector 
                    running_loss  = 0               # Initialize loss vector
                    for j in range(0,NB):           # For each batch of proteins
                                                       
                        lim_inf  = j*Real_batch                                          # Calculate lower limit
                        lim_sup  = (j+1)*Real_batch                                      # Calculate upper limit
                        Z_repeat = Z_i.repeat(Real_batch,1).unsqueeze(0).permute(1,0,2)  # Repeat ith protein embedding Real_batch times
                        Zj       = Z[:,lim_inf:lim_sup,:].permute(1,0,2)                 # Get   the jth batch of protein embeddings
                        Xb       = torch.cat((Z_repeat ,Zj ),axis=1)                     # Concatenate the two tensors
                        yb       =  Y_i[:,lim_inf:lim_sup].squeeze()                     # Get the jth batch of labels

                        y_hat  = model(Xb)                                              # Forward pass
                        y_prob = activation_function(y_hat)                             # Get probability of interaction
                        y_pred = (y_prob > threshold).float()                           # Get prediction
                        acc    = (y_pred == yb).float().mean()                          # Calculate accuracy
                        accuracy_list = np.append(accuracy_list,acc)                    # Add accuracy to accuracy list
                accuracy_list_epoch = np.append(accuracy_list_epoch,np.mean(accuracy_list))         # Add average accuracy to accuracy list

        return accuracy_list_epoch
                

                
def inference(model,X,activation_function = nn.Sigmoid(),threshold:float=0.5 ):
    model.eval()
    with torch.no_grad():
        y_hat  = model(X)
        y_prob = activation_function(y_hat)
        y_pred = (y_prob > threshold).float()
        return (y_prob , y_pred)
   
    