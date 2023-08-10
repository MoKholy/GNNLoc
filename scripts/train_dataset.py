from accelerate import Accelerator
from train import load_dataset
from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np
from model import GCN
import argparse
import warnings
warnings.filterwarnings("ignore")


# training function
def train(model, name, train_loader, val_loader, epochs, lr=0.001, mae_loss=True):

    accelerator = Accelerator()
    # set optimizer and loss function 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # loss functions
    mae = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()

    # accelerator 
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    train_loss_epoch_mae = []
    train_loss_epoch_mse = []
    val_loss_epoch_mae = []
    val_loss_epoch_mse = []
    for epoch in range(epochs):

        # set model to train
        model.train()

        # collect losses
        train_losses_mae = []
        train_losses_mse = []

        # set tqdm loop 
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

        for batch_idx, data in loop:

            # get data
            x, edge_attr, edge_index, batch_index, y = data.x, data.edge_attr, data.edge_index, data.batch, data.y

            # reshape y
            # y = y.reshape(-1, 2)

            # get output
            output = model(x, edge_attr, edge_index, batch_index)

            # print(output.shape, y.shape)
            # calculate loss for x and y  dims
            if output.shape != y.shape:
                continue
            loss_mae = mae(output[:, :2], y[:, :2])
            loss_mse = mse(output[:, :2], y[:, :2])

            # collect losses
            train_losses_mae.append(loss_mae.item())
            train_losses_mse.append(loss_mse.item())

            # zero grad
            optimizer.zero_grad()

            # backpropagate
            if mae_loss:
                accelerator.backward(loss_mae)
            else:
                accelerator.backward(loss_mse)

            # optimizer step
            optimizer.step()

            # set description for tqdm
            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(loss_mae=loss_mae.item(), loss_mse=loss_mse.item())
        
        # get avg losses
        avg_train_mae = np.mean(train_losses_mae)
        avg_train_mse = np.mean(train_losses_mse)

        # append to epoch losses
        train_loss_epoch_mae.append(avg_train_mae)
        train_loss_epoch_mse.append(avg_train_mse)

        # evaluate on validation set
        model.eval()

        # collect losses
        val_losses_mae = []
        val_losses_mse = []

        # set tqdm loop
        val_loop = tqdm(enumerate(val_loader), total=len(val_loader))
        
        # torch no grad
        with torch.no_grad():
            for batch_idx, data in val_loop:

                # get data
                x, edge_attr, edge_index, batch_index, y = data.x, data.edge_attr, data.edge_index, data.batch, data.y

                # # reshape y
                # y = y.reshape(-1, 2)

                # get output
                output = model(x, edge_attr, edge_index, batch_index)
                if output.shape != y.shape:
                    continue
                # assert output.shape == y.shape
                # calculate loss for x and y dims
                loss_mae = mae(output[:, :2], y[:, :2])
                loss_mse = mse(output[:, :2], y[:, :2])

                # collect losses
                val_losses_mae.append(loss_mae.item())
                val_losses_mse.append(loss_mse.item())

                # set description for tqdm
                val_loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
                val_loop.set_postfix(avg_train_loss_mae = avg_train_mae, avg_train_loss_mse = avg_train_mse, val_loss_mae=loss_mae.item(), val_loss_mse=loss_mse.item())

        # get avg losses
        avg_val_mae = np.mean(val_losses_mae)
        avg_val_mse = np.mean(val_losses_mse)

        # append to epoch losses
        val_loss_epoch_mae.append(avg_val_mae)
        val_loss_epoch_mse.append(avg_val_mse)

    # wait for accelerator to finish
    accelerator.wait_for_everyone()

    # save model if main process
    if accelerator.is_main_process:
        torch.save(model.state_dict(), f"../weights/{name}.pt")

    # return losses
    return train_loss_epoch_mae, train_loss_epoch_mse, val_loss_epoch_mae, val_loss_epoch_mse


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mae", type=bool, default=False)
    parser.add_argument("--trainratio", type=float, default=0.8)
    args = parser.parse_args()

    # load dataset
    train_loader, val_loader = load_dataset(name=args.dataset, k=args.k, batch_size=args.batchsize, train_ratio=args.trainratio, test_ratio=(1-args.trainratio))

    # create model
    model = GCN(3, 32, 128, 2, 2, 0.3)

    # train model
    train_loss_epoch_mae, train_loss_epoch_mse, val_loss_epoch_mae, val_loss_epoch_mse = train(model, args.name, train_loader, val_loader, args.epochs, lr=args.lr, mae_loss=args.mae)

    # print final losses
    print(f"Train Loss MAE: {train_loss_epoch_mae[-1]}")
    print(f"Train Loss MSE: {train_loss_epoch_mse[-1]}")
    print(f"- - - - - - - - - - - -- - - -")
    print(f"Val Loss MAE: {val_loss_epoch_mae[-1]}")
    print(f"Val Loss MSE: {val_loss_epoch_mse[-1]}")

