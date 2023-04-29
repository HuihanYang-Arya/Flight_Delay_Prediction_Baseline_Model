import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from GRU import GRUNet
import util
import argparse
import random
import copy
import numpy as np
from baseline_methods import test_error, StandardScaler

import json
import argparse
import pandas as pd
# Read the configuration file

def store_result(yhat,label,i):
    metrics = test_error(yhat[:,:,i],label[:,:,i])
    return metrics[0],metrics[2],metrics[1]

def main(delay_index = 0):
    with open('configs.json', 'r') as f:
        config = json.load(f)

    # Create an empty argparse Namespace object to store the configuration settings
    args = argparse.Namespace()
    for key, value in config.items():
        setattr(args, key, value)
        device = torch.device(args.device)

    device = torch.device(args.device)

    # Load your dataset
    adj, training_data, val_data, training_w, val_w = util.load_data(args.data)
    adj = adj[1]

    model = GRUNet(in_c=2, hid_c=args.lstm_hidden_size, out_c=1).to(device)
    optimizer, scheduler, scaler, training_data,batch_index,val_index = util.model_preprocess(model,args.lr,args.gamma,args.step_size,training_data,val_data,args.in_len,args.out_len)
    label=util.label_loader(val_index,args.in_len,args.out_len,delay_index,val_data)
    n_epochs = args.episode
    # Train the model
    print("start training...",flush=True)
    amae3, ar3, armser3, amae6, ar6, armser6, amae12, ar12, armser12  = [],[],[],[],[],[],[],[],[]
    for ep in range(n_epochs):
        model.train()
        random.shuffle(batch_index)
        for j in range(len(batch_index) // args.batch - 1):
            trainx, trainy,trainw = util.train_dataloader(batch_index, args.batch, training_data, training_w,j,args.in_len, args.out_len)
            trainw = torch.LongTensor(trainw).to(device)
            trainx =  torch.index_select(torch.Tensor(trainx), -1, torch.tensor([delay_index]))
            trainx = trainx.to(device)
            trainw = trainw.unsqueeze(-1)
            train = torch.cat((trainw, trainx), dim=-1)
            trainy = torch.index_select(torch.Tensor(trainy), -1, torch.tensor([delay_index]))
            trainy = trainy.to(device)
            trainy = trainy.permute(0, 3, 1, 2)[:,0,:,:]
            data = {"flow_x": train, "graph": adj}
            optimizer.zero_grad()
            output = model(data,device)
            output = output.squeeze()
            loss = util.masked_rmse(output, trainy, 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

        scheduler.step()     
        outputs = []

        # Evaluate the model
        model.eval()
        print('evaluating')
        for i in range(len(val_index)):
            testx, testw = util.test_dataloader(val_index, val_data, val_w,i,args.in_len, args.out_len)
            testx = scaler.transform(testx)
            testw = torch.LongTensor(testw).to(device)
            testx[np.isnan(testx)] = 0
            testx =  torch.index_select(torch.Tensor(testx), -1, torch.tensor([delay_index]))
            testx = testx.to(device)
            testw = testw.unsqueeze(-1)
            test = torch.cat((testw, testx), dim=-1)
            data = {"flow_x": test, "graph": adj}
            output = model(data,device)
            output = output.detach().cpu().numpy()
            output = scaler.inverse_transform(output)
            outputs.append(output)
        yhat = np.concatenate(outputs)
        amae = []
        ar2 = []
        armse = []
        for i in range(2):
            metrics = test_error(yhat[:,:,i,0],label[:,:,i,0])
            amae.append(metrics[0])
            ar2.append(metrics[2])
            armse.append(metrics[1])
         
        log = 'On average over all horizons, Test MAE: {:.4f}, Test R2: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(amae),np.mean(ar2),np.mean(armse)))

if __name__ == '__main__':
    main()