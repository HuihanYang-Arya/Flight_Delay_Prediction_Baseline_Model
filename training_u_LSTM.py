import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from LSTM import EncoderDecoder
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
from torch.utils.tensorboard import SummaryWriter

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
    model = EncoderDecoder(input_size=1, hidden_size=args.lstm_hidden_size, output_size=1).to(device)
    optimizer, scheduler, scaler, training_data,batch_index,val_index = util.model_preprocess(model,args.lr,args.gamma,args.step_size,training_data,val_data,args.in_len,args.out_len)
    label=util.label_loader(val_index,args.in_len,args.out_len,delay_index,val_data)
    n_epochs = args.episode
    # Train the model
    print("start training...",flush=True)
    writer = SummaryWriter('114154')
    amae3, ar3, armser3, amae6, ar6, armser6, amae12, ar12, armser12  = [],[],[],[],[],[],[],[],[]
    for ep in range(n_epochs):
        random.shuffle(batch_index)
        a = len(batch_index) // args.batch - 1
        for j in range(len(batch_index) // args.batch - 1):
            
            trainx, trainy,trainw = util.train_dataloader(batch_index, args.batch, training_data, training_w,j,args.in_len, args.out_len)
            trainw = torch.LongTensor(trainw).to(device)
            #只做了第一个的延迟
            trainx =  torch.index_select(torch.Tensor(trainx), -1, torch.tensor([delay_index]))
            # Combine the data of all airports into a single sequence for each sample
            trainx = trainx.view(-1, args.batch, 12*1)
            # Permute the dimensions to have the batch size as the first dimension
            trainx = trainx.permute(1, 0, 2)
            trainx = trainx.to(device)
            trainy = torch.index_select(torch.Tensor(trainy), -1, torch.tensor([delay_index]))
            #trainy = torch.Tensor(trainy)
            trainy = trainy.to(device)
            #trainy = trainy.permute(0, 3, 1, 2)
            model.train()
            optimizer.zero_grad()
            output = model(trainx,12)
            loss = util.masked_mae(output, trainy, 0.0)
            loss.backward()
            writer.add_scalar('training loss',loss,ep*a+j)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
        scheduler.step()     
        outputs = []

        # Evaluate the model
        if ep %3 == 0:
            model.eval()
            print('evaluating')
            for i in range(len(val_index)):
                testx, testw = util.test_dataloader(val_index, val_data, val_w,i,args.in_len, args.out_len)
                testx = scaler.transform(testx)
                testx[np.isnan(testx)] = 0
                testx =  torch.index_select(torch.Tensor(testx), -1, torch.tensor([delay_index]))
                testx = testx.to(device)
                output = model(testx,12)
                output = output.detach().cpu().numpy()
                output = scaler.inverse_transform(output)
                outputs.append(output)
            yhat = np.concatenate(outputs)
            prediction2 = store_result(yhat,label,2)
            amae3.append(prediction2[0])
            ar3.append(prediction2[1])
            armser3.append(prediction2[2])
            print(ar3,armser3)
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
    df.to_csv('./LSTM.csv')

if __name__ == '__main__':
    main()