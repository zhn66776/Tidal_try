from statistics import mean, mode
from tqdm import tqdm
from numpy import sqrt
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import torch
def CNNBiLstm_evaluate(model, loader, config, val_mode=False):
    model.eval()

    y = []
    y_pre = []

    with torch.no_grad():
      for idx, (X, Y) in tqdm(enumerate(loader)):
            # Conv1d接受的数据输入是(batch_size有, channel_size=1, seq_len有),故增加一个通道数
            X = X.unsqueeze(-2).to(config.device)
            Y = Y.to(config.device)

            predictions = []
            input_seq = X.clone()  # 复制当前输入，作为后续预测的初始输入

            for _ in range(config.output_size):  # 根据所需的预测步数（例如5步）
                pred = model(input_seq)  # 用当前输入得到预测值
                predictions.append(pred.cpu().squeeze().tolist())  # 将预测值保存到列表

                # 将预测值作为下一次输入
                pred = pred.unsqueeze(-2)  # 添加通道维度以符合输入要求
                input_seq = torch.cat((input_seq[:, :, 1:], pred), dim=2)  # 用预测值更新输入序列
            for i in range (len(predictions)):
              y_pre.append(predictions[i])
              y.append(Y[:, i].cpu().squeeze().tolist())

            # 将预测结果列表展开，并添加到 y_pred
            y_pre += [item for sublist in y_pre for item in sublist]
            y += [item for sublist in y for item in sublist]
            if val_mode:
              valmeanSquaredError = mean_squared_error(y_true=y, y_pred=y_pre)
              return valmeanSquaredError
            else:
              r2Score = r2_score(y_true=y, y_pred=y_pre)
              meanSquaredError = mean_squared_error(y_true=y, y_pred=y_pre)
              meanAbsoluteError = mean_absolute_error(y_true=y, y_pred=y_pre)
              print("r2Score: ", r2Score)
              print("meanSquaredError: ", meanSquaredError)
              print('RMSE: ',sqrt(meanSquaredError))
              print("meanAbsoluteError: ", meanAbsoluteError)

        # 画出实际结果和预测的结果
              import matplotlib.pyplot as plt
              plt.plot(range(len(y[:1000])),y_pre[:1000],color = 'red',linewidth = 1.5,linestyle = '-.',label='prediction')
              plt.plot(range(len(y[:1000])),y[:1000],color = 'blue',linewidth = 1.5,linestyle = '-', label='real')
              plt.legend(loc='best')

              return y_pre,y

def CNNBiLstmtrain(model, trainloader,valloader, criterion, optimizer, config):
    logfile = open("./log/trainLog.txt",mode='a+',encoding='utf-8')
    for epoch in range(config.epoch_size):
        model.train()
        for idx, (X, Y) in enumerate(trainloader):
            # Conv1d接受的数据输入是(batch_size有, channel_size=1, seq_len有),故增加一个通道数，单序列通道数为1，第2维
            X = X.unsqueeze(-2).to(config.device)
            Y = Y.to(config.device)
            
            predict = model(X)
            loss = criterion(predict, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f"Epoch: {epoch} batch: {idx} | loss: {loss}")
                
        
        # 一个epoch结束,进行一次验证,并保存相关记录
        valMSE = CNNBiLstm_evaluate(model,loader = valloader,config=config,val_mode=True)
        print(f"Epoch: {epoch} valLoss: {valMSE}")
        logfile.write("Epoch: {0} valLoss: {1} \n".format(epoch,valMSE))

    logfile.write("\n")
    logfile.close()
    return model
