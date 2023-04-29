import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from GAT import GATNet
import util
import argparse
import random
import numpy as np
from baseline_methods import test_error, StandardScaler
import json
import argparse
import pandas as pd

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
    adj, training_data, val_data, training_w, val_w = util.load_data(args.data)
    adj = adj[0]
    model = GATNet(in_c=24, hid_c=12, out_c=12, n_heads=4).to(device)
    optimizer, scheduler, scaler, training_data,batch_index,val_index = util.model_preprocess(model,args.lr,args.gamma,args.step_size,training_data,val_data,args.in_len,args.out_len)
    label=util.label_loader(val_index,args.in_len,args.out_len,delay_index,val_data)
    label=util.label_loader(val_index,args.in_len,args.out_len,delay_index,val_data,graph = True)

    # Train the model
    print("start training...",flush=True)
    amae3, ar3, armser3, amae6, ar6, armser6, amae12, ar12, armser12  = [],[],[],[],[],[],[],[],[]    
    for ep in range(1,1+args.episode):
        model.train()
        random.shuffle(batch_index)
        for j in range(len(batch_index) // args.batch - 1):
            trainx, trainy,trainw = util.train_dataloader(batch_index, args.batch, training_data, training_w,j,args.in_len, args.out_len)
            trainx =  torch.index_select(torch.Tensor(trainx), -1, torch.tensor([delay_index]))
            trainx = trainx.to(device)
            trainw = torch.LongTensor(trainw).to(device)
            trainw = trainw.unsqueeze(-1)
            trainy = torch.index_select(torch.Tensor(trainy), -1, torch.tensor([delay_index]))
            trainy = trainy.to(device)[:,:,:,0]
            train = torch.cat((trainw, trainx), dim=-1)
            data = {"flow_x": train, "graph": adj}
            optimizer.zero_grad()
            output = model(data,device)
            loss = util.masked_rmse(output, trainy, 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            
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
            output = model(data,device).squeeze(dim=2)
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
    df.to_csv('./baseline_results/GAT.csv')
    
if __name__ == "__main__":   
    main()  