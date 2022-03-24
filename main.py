import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from stagcn import STAGCN
from utils import generate_dataset,  load_metr_la_data4, load_metr_la_data8
from libs.metrics import RMSE, masked_mape_np, MAE

use_gpu = True


Ks, Kt = 3, 3
blocks = [[1, 32, 64], [64, 32, 128]]
drop_prob = 0
n_days = 288
num_timesteps_input = 12
num_timesteps_output = 12

epochs = 3
batch_size = 50

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples,num_features,num_timesteps_train,num_nodes).
    :param training_target: Training targets of shape (num_samples, num_nodes,num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    # disrupt the order of the original data
    permutation = torch.randperm(training_input.shape[0])
    epoch_training_losses = []
    for i in range(0, training_input.shape[0]-batch_size+1, batch_size):
        net.train()
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)
        out = net(X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())  #loss.detach().cpu().numpy()
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':
    # set random factor
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    A, X, means, stds = load_metr_la_data8()
    # segmentation data
    n_route = X.shape[0]
    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)
    # train_original_data (num_nodes,  num_features, num_samples)
    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]
    # training_input.shape[20549, 207, 12, 2]，test_target[6841, 207, 3]
    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)

    # initialization model
    net = STAGCN(Ks, Kt, blocks, num_timesteps_input, n_route, drop_prob, num_timesteps_output, training_input.shape[1], A, n_days).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    testidation_losses = []
    testidation_maes = []
    testidation_mape = []
    testidation_rmse = []

    for epoch in range(epochs):
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        training_losses.append(loss)

        # Run validation
        with torch.no_grad():
            net.eval()
            # val
            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)
            # no disruption of data order
            permutation = torch.arange(0, val_input.shape[0])
            val_last = permutation[0:int(val_input.shape[0] / 50) * 50]
            epoch_val_losses = []
            for i in range(0, val_input.shape[0] - batch_size + 1, batch_size):
                indices = permutation[i:i + batch_size]  # 1-d array of shape 50

                val_input_batch, val_target_batch = val_input[indices], val_target[indices]
                val_input_batch = val_input_batch.to(device=args.device)
                val_target_batch = val_target_batch.to(device=args.device)
                out = net(val_input_batch)
                # save the results to out_val
                if i == 0:
                    out_last = out
                elif i > 0:
                    out_val = torch.cat((out_last, out), 0)
                    out_last = out_val
                # calculate loss value
                loss2 = loss_criterion(out, val_target_batch).to(device='cuda')

                epoch_val_losses.append(
                    loss2.detach().cpu().numpy())
            val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
            validation_losses.append(val_loss)

            # start test
            test_input = test_input.to(device=args.device)
            test_target = test_target.to(device=args.device)
            # no disruption of data order
            permutation = torch.arange(0, test_input.shape[0])
            test_last = permutation[0:int(test_input.shape[0] / 50) * 50]
            epoch_test_losses = []
            # out_test = []
            for i in range(0, test_input.shape[0] - batch_size + 1, batch_size):
                indices = permutation[i:i + batch_size]

                test_input_batch, test_target_batch = test_input[indices], test_target[indices]
                test_input_batch = test_input_batch.to(device=args.device)
                test_target_batch = test_target_batch.to(device=args.device)
                out = net(test_input_batch)
                # save the results to out_val
                if i == 0:
                    out_last = out
                elif i > 0:
                    out_test = torch.cat((out_last, out), 0)
                    out_last = out_test

                # calculate loss value
                loss2 = loss_criterion(out, test_target_batch).to(device='cuda')
                epoch_test_losses.append(
                    loss2.detach().cpu().numpy())  # loss不仅有损失值还有其它一些信息，loss.detach().cpu().numpy()是只取损失值
            test_loss = sum(epoch_test_losses) / len(epoch_test_losses)
            testidation_losses.append(test_loss)
            test_target_out = test_target[test_last]
            # evaluation test effect
            out_unnormalized = out_test.detach().cpu().numpy() * stds[0] + means[0]
            target_unnormalized = test_target_out.detach().cpu().numpy() * stds[0] + means[0]
            mae = MAE(out_unnormalized, target_unnormalized)
            testidation_maes.append(mae)

            mape = masked_mape_np(out_unnormalized, target_unnormalized, 0)
            testidation_mape.append(mape)

            rmse = RMSE(out_unnormalized, target_unnormalized)
            testidation_rmse.append(rmse)
            out = None
            test_input = test_input.to(device=args.device)
            test_target = test_target.to(device=args.device)

            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)

        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Testidation loss: {}".format(testidation_losses[-1]))
        print("Testidation MAE: {}".format(testidation_maes[-1]))
        print("Testidation MAPE: {}".format(testidation_mape[-1]))
        print("Testidation RSME: {}".format(testidation_rmse[-1]))
        print("Testidation MAEmean: {}".format(np.mean(testidation_maes)))
        print("Testidation MAPEmean: {}".format(np.mean(testidation_mape)))
        print("Testidation RSMEmean: {}".format(np.mean(testidation_rmse)))

        checkpoint_path = "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open("checkpoints/losses.pk", "wb") as fd:
            pk.dump((training_losses, validation_losses, testidation_losses, testidation_maes, testidation_mape,
                     testidation_rmse), fd)
