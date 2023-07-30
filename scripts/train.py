from tqdm import tqdm
import torch
import torch.optim as optim
from accelerate import Accelerator
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
import argparse
import os
from utils import *
from dataset import Dataset
from model import GCN
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter

# function to parse arguments
def parse_args():
    pass

# function to load dataset
def load_dataset(name, k, batch_size, train_ratio, test_ratio):
    # get dataset
    dataset = Dataset(root="./data", name=name, k=k)
    # get train and test size
    train_size = int(train_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    # get train and test dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # get train and test loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# train model, report r2 score, MAE and loss to summary writer without function
def train_model(model, name, train_loader, test_loader, epochs, accelerator, writer_dir, lr=0.001):
    
    # set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # mae for loss, mse and r2 for evaluation, all reported
    mae = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()

    # writer for tensorboard
    writer = SummaryWriter(writer_dir)

    # accelerator
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    # train loop
    for epoch in tqdm(range(epochs)):

        # set model to train
        model.train()

        # collect losses
        train_losses_mae = []
        train_losses_mse = []

        for data in train_loader:

            # get data
            x, edge_attr, edge_index, batch_index, y = data.x, data.edge_attr, data.edge_index, data.batch, data.y

            # reshape y
            y = y.reshape(-1, 3)

            # get output
            output = model(x, edge_attr, edge_index, batch_index)

            # calculate loss for x and y  dims
            loss_mae = mae(output[:, :2], y[:, :2])
            loss_mse = mse(output[:, :2], y[:, :2])

            # collect losses
            train_losses_mae.append(accelerator.gather(loss_mae.item()))
            train_losses_mse.append(accelerator.gather(loss_mse.item()))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

            # zero grad                                                                                                                                                                                                                                                                                                                             
            optimizer.zero_grad()

            # backpropagate                             
            accelerator.backward(loss_mae)                                                  

            # step optimizer
            optimizer.step()

        # get avg traiin loss
        avg_train_mae = np.mean(train_losses_mae)
        avg_train_mse = np.mean(train_losses_mse)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train MSE: {avg_train_mse:.4f}, Train MAE: {avg_train_mae:.4f}")

        # log to tensorboard
        if accelerator.is_main_process:
            writer.add_scalar("Train MSE", avg_train_mse, epoch)
            writer.add_scalar("Train MAE", avg_train_mae, epoch)

        # set model to eval
        model.eval()

        # collect losses
        test_losses_mae = []
        test_losses_mse = []
        test_r2_scores = []

        # no grad
        with torch.no_grad():
            for data in test_loader:

                # get data
                x, edge_attr, edge_index, batch_index, y = data.x, data.edge_attr, data.edge_index, data.batch, data.y

                # reshape y
                y = y.reshape(-1, 3)

                # get output
                output = model(x, edge_attr, edge_index, batch_index)

                # calculate loss for x and y  dims
                loss_mae = mae(output[:, :2], y[:, :2])
                loss_mse = mse(output[:, :2], y[:, :2])

                # collect losses
                test_losses_mae.append(accelerator.gather(loss_mae.item()))
                test_losses_mse.append(accelerator.gather(loss_mse.item()))

                # get r2 score
                r2 = r2_score(y[:, :2].cpu().numpy(), output[:, :2].cpu().numpy())
                test_r2_scores.append(r2)

            # get avg test loss
            avg_test_mae = np.mean(test_losses_mae)
            avg_test_mse = np.mean(test_losses_mse)
            avg_test_r2 = np.mean(test_r2_scores)

            print(f"Test MSE: {avg_test_mse:.4f}, Test MAE: {avg_test_mae:.4f}, Test R2: {avg_test_r2:.4f}")

            # log to tensorboard
            if accelerator.is_main_process:
                writer.add_scalar(f"{name}_Test MSE", avg_test_mse, epoch)
                writer.add_scalar(f"{name}_Test MAE", avg_test_mae, epoch)
                writer.add_scalar(f"{name}_Test R2", avg_test_r2, epoch)

    # close writer
    writer.close()

    # save model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # save model
        accelerator.save_model(model, f"./weights/{name}.pt")




























