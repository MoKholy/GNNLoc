import torch 
from accelerate import Accelerator
from dataset import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from model import GCN, GAT, GAT2
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore")


# evaluate function

def evaluate(model, loader, dataset):
    accelerator = Accelerator()
    # prepare accelerator and model
    model, loader = accelerator.prepare(model, loader)

    # set model to eval
    model.eval()
    mae_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()
    # collect losses
    losses_mae = []
    losses_mse = []
    # torch no grad
    with torch.no_grad():

        for data in loader:

            x, edge_attr, edge_index, batch_index, y = data.x, data.edge_attr, data.edge_index, data.batch, data.y

            # get indices for denormalization
            indices = data.indx.cpu().numpy()
            # print(indices)
            # get output
            # print(indices)
            output = model(x, edge_attr, edge_index, batch_index).cpu().numpy()
            # print(output)
            for i in range(len(output)):
                
                # get index
                index = indices[i]

                # get mapping
                # if index not in dataset.map:
                #     continue
                # else:
                #     
                curr_mapping = dataset.get_mapping(index)

                # denormalize output

                original_x, original_y= curr_mapping[0]
                strongest_ap_x, strongest_ap_y = curr_mapping[2]
                
                denormalized_x, denormalized_y = output[i, 0] + strongest_ap_x, output[i, 1] + strongest_ap_y
                
                # get loss for x and y dims

                # calculate mae and mse loss
                loss_mae = mae_loss(torch.tensor([denormalized_x, denormalized_y]), torch.tensor([original_x, original_y]))
                loss_mse = mse_loss(torch.tensor([denormalized_x, denormalized_y]), torch.tensor([original_x, original_y]))
                
                # append losses
                losses_mae.append(loss_mae.item())
                losses_mse.append(loss_mse.item())

    # calculate mean losses
    avg_loss_mae = np.mean(losses_mae)
    avg_loss_mse = np.mean(losses_mse)

    # sort losses
    losses_mae = sorted(losses_mae)
    losses_mse = sorted(losses_mse)
    # get median 
    median_loss_mae = losses_mae[len(losses_mae)//2]
    median_loss_mse = losses_mse[len(losses_mse)//2]

    print(f"AVG Test MSE: {avg_loss_mse:.2f}, AVG Test MAE: {avg_loss_mae:.2f}")
    print(f"Median Test MSE: {median_loss_mse:.2f}, Median Test MAE: {median_loss_mae:.2f}")

    # print max
    print(f"Max Test MSE: {max(losses_mse):.2f}, Max Test MAE: {max(losses_mae):.2f}")
    # print min
    print(f"Min Test MSE: {min(losses_mse):.2f}, Min Test MAE: {min(losses_mae):.2f}")
    return avg_loss_mae, avg_loss_mse

if __name__ == "__main__":

    # parse args
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    # load dataset
    dataset = Dataset(root="../data", name=args.dataset, k=args.k)
    test_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=16) 

    # load model
    # model = GAT2(3, 128, 256, 2, 2, 0.3, heads=2) # model 1
    # model = GAT2(3, 128, 256, 2, 2, 0.3, heads=4) # model 2
    model = GAT2(3, 128, 256, 2, 2, 0.3, heads=6) # model 3
    # model = GAT2(3, 256, 256, 2, 2, 0.3, heads=4) # model 4
    model.load_state_dict(torch.load(f"../weights/{args.model_name}.pt"))

    # evaluate
    avg_loss_mae, avg_loss_mse = evaluate(model, test_loader, dataset)

    # print results
    # print(f"Test MSE: {avg_loss_mse:.4f}, Test MAE: {avg_loss_mae:.4f}")
