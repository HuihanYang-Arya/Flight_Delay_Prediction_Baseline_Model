import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from ASTGCN_r import make_model
import util
import random
import numpy as np
from baseline_methods import test_error, StandardScaler
import pandas as pd
import json
import argparse


def store_result(yhat,label,i):
    metrics = test_error(yhat[:,:,i],label[:,:,i])
    return metrics[0],metrics[2],metrics[1]

def main(delay_index = 0):
    # Read the configuration file
    with open('configs.json', 'r') as f:
        config = json.load(f)

    # Create an empty argparse Namespace object to store the configuration settings
    args = argparse.Namespace()
    for key, value in config.items():
        setattr(args, key, value)
        device = torch.device(args.device)

    # Load your dataset
    adj, training_data, val_data, training_w, val_w = util.load_data(args.data,ratio=args.train_val_ratio)
    adj = adj[1]
    model = make_model('cuda:0', nb_block=2, in_channels=2, K=3, nb_chev_filter=3, nb_time_filter=3, time_strides=3, adj_mx=adj, num_for_predict=args.out_len, len_input=args.in_len, num_of_vertices=70)
    optimizer, scheduler, scaler, training_data,batch_index,val_index = util.model_preprocess(model,args.lr,args.gamma,args.step_size,training_data,val_data,args.in_len,args.out_len)
    label=util.label_loader(val_index,args.in_len,args.out_len,delay_index,val_data,graph = True)
    n_epochs = args.episode
    # Train the model
    print("start training...",flush=True)
    amae3, ar3, armser3, amae6, ar6, armser6, amae12, ar12, armser12  = [],[],[],[],[],[],[],[],[]
    for ep in range(n_epochs):
        random.shuffle(batch_index)
        model.train()
        for j in range(len(batch_index) // args.batch - 1):
            trainx, trainy,trainw = util.train_dataloader(batch_index, args.batch, training_data, training_w,j,args.in_len, args.out_len)
            trainw = torch.LongTensor(trainw).to(device)
            trainx =  torch.index_select(torch.Tensor(trainx), -1, torch.tensor([delay_index])).to(device)
            trainw = trainw.unsqueeze(-1)
            trainy = torch.index_select(torch.Tensor(trainy), -1, torch.tensor([delay_index])).to(device)
            trainy = trainy.permute(0, 3, 1, 2)
            trainy = trainy.squeeze(dim=1)
            train = torch.cat((trainw, trainx), dim=-1).permute(0, 1, 3, 2)
            optimizer.zero_grad()
            output = model(train)
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
            testx[np.isnan(testx)] = 0
            testx =  torch.index_select(torch.Tensor(testx), -1, torch.tensor([delay_index]))
            testx = testx.to(device)
            testw = torch.LongTensor(testw).to(device)
            testw = testw.unsqueeze(-1)
            test = torch.cat((testw, testx), dim=-1).permute(0, 1, 3, 2)
            output = model(test)
            output = output.detach().cpu().numpy()
            output = scaler.inverse_transform(output)
            outputs.append(output)
        yhat = np.concatenate(outputs)
        prediction2 = store_result(yhat,label,2)
        amae3.append(prediction2[0])
        ar3.append(prediction2[1])
        armser3.append(prediction2[2])
        print(ar3)
        prediction5 = store_result(yhat,label,5)
        amae6.append(prediction5[0])
        ar6.append(prediction5[1])
        armser6.append(prediction5[2])
        prediction11 = store_result(yhat,label,11)
        amae12.append(prediction11[0])
        ar12.append(prediction11[1])
        armser12.append(prediction11[2])
         
    df = pd.DataFrame()
    df['amae3'] = amae3
    df['ar3'] = ar3
    df['armser3'] = armser3
    df['amae6'] = amae6
    df['ar6'] = ar6
    df['armser6'] = armser6
    df['amae12'] = amae12
    df['ar12'] = ar12
    df['armser12'] = armser12
    df.to_csv('./baseline_results/astgcn.csv')

if __name__ == "__main__":   
    main()  













