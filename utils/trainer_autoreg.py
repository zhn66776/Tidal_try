from statistics import mean, mode
from tqdm import tqdm
from numpy import sqrt
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import torch
import numpy as np


    
###111用预测的去预测凌晨evaluate#############################################################################
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

def CNNBiLstm_evaluate(model, initial_sequence, test_Y, config):
    model.eval()
    
    y_true = test_Y.squeeze().tolist()  # 真实的标签值列表
    y_pred = []  # 存储模型的预测值

    
    current_sequence = initial_sequence.squeeze().tolist()

    # 预测步数，与测试集的长度一致
    predict_steps = len(y_true)

    for i in range(predict_steps):
        
        input_seq = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-2).to(config.device)
        with torch.no_grad():
            predicted_value = model(input_seq).cpu().item()
        
        # 保存预测结果
        y_pred.append(predicted_value)
        
        # 更新输入序列：移除第一个值，添加预测的值
        current_sequence.pop(0)
        current_sequence.append(predicted_value)

    # 计算评估指标
    r2Score = r2_score(y_true=y_true, y_pred=y_pred)
    meanSquaredError = mean_squared_error(y_true=y_true, y_pred=y_pred)
    meanAbsoluteError = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    print("r2Score: ", r2Score)
    print("meanSquaredError: ", meanSquaredError)
    print('RMSE: ', sqrt(meanSquaredError))
    print("meanAbsoluteError: ", meanAbsoluteError)

    # 绘制预测结果和真实值的对比图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 3))
    plt.plot(range(len(y_true)), y_pred, color='red', linewidth=1.5, linestyle='-.', label='Prediction')
    plt.plot(range(len(y_true)), y_true, color='blue', linewidth=1.5, linestyle='-', label='Real')
    plt.legend(loc='best')
    plt.show()

    return y_pred, y_true


#11用预测凌晨train##################################################################################################################
def CNNBiLstmtrain(model, trainloader, valloader, criterion, optimizer, config):
    logfile = open("./log/trainLog.txt", mode='a+', encoding='utf-8')
    for epoch in range(config.epoch_size):
        model.train()
        for idx, (X, Y) in enumerate(trainloader):
            # Conv1d接受的数据输入是(batch_size, channel_size=1, seq_len)，故增加一个通道数，单序列通道数为1，第2维
            X = X.unsqueeze(-2).to(config.device)
            Y = Y.to(config.device)
            
            predict = model(X)
            loss = criterion(predict, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f"Epoch: {epoch} batch: {idx} | loss: {loss.item()}")
                
        # 一个epoch结束,进行一次验证,并保存相关记录
        valMSE = CNNBiLstm_evaluate_validation(model, valloader, config)
        print(f"Epoch: {epoch} valLoss: {valMSE}")
        logfile.write("Epoch: {0} valLoss: {1} \n".format(epoch, valMSE))

    logfile.write("\n")
    logfile.close()
    return model

def CNNBiLstm_evaluate_validation(model, loader, config):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for idx, (X, Y) in enumerate(loader):
            X = X.unsqueeze(-2).to(config.device)
            Y = Y.to(config.device)

            output = model(X)
            y_pred.extend(output.cpu().squeeze().tolist())
            y_true.extend(Y.cpu().squeeze().tolist())

    valMSE = mean_squared_error(y_true=y_true, y_pred=y_pred)
    return valMSE


def RNNtrain(model, trainloader,valloader,criterion, optimizer, config):
    logfile = open("./log/trainLog.txt",mode='a+',encoding='utf-8')
    for epoch in range(config.epoch_size):
        # 每个epoch开始模型训练模式
        model.train()
        for idx, (X, Y) in enumerate(trainloader):
            # batch first,(batch,seq_len,input_size),所以增加第三个维度.因为单序列,所以inputsize维度为1
            X = X.unsqueeze(2).to(config.device)
            Y = Y.to(config.device)
            
            predict = model(X)
            loss = criterion(predict, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f"Epoch: {epoch} batch: {idx} | loss: {loss}")
                
        
        # 一个epoch结束,进行一次验证,并保存相关记录
        valMSE = RNNevaluate(model,loader = valloader,config=config,val_mode=True)
        print(f"Epoch: {epoch} valLoss: {valMSE}")
        logfile.write("Epoch: {0} valLoss: {1} \n".format(epoch,valMSE))
    logfile.write("\n")
    logfile.close()
    return model
