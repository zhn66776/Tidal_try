from statistics import mean, mode
from tqdm import tqdm
from numpy import sqrt
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def RNNevaluate(model, loader, config, val_mode=False):
    model.eval()

    y = list()
    y_pre = list()
    for idx, (X, Y) in tqdm(enumerate(loader)):
        # batch first,(batch,seq_len,input_size),所以增加第三个维度.因为单序列,所以inputsize维度为1
        X = X.unsqueeze(2).to(config.device)
        Y = Y.to(config.device)
        
        y_pre += model(X).cpu().squeeze().tolist()
        y += Y.cpu().squeeze().tolist()

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
##用预测的值来预测
def CNNBiLstm_evaluate(model, loader, config, val_mode=False):
    model.eval()
    y_true = []
    y_pred = []
    n_steps = config.n_steps

    with torch.no_grad():
        for idx, (X, Y) in tqdm(enumerate(loader)):
            X = X.unsqueeze(-2).to(config.device)
            Y = Y.to(config.device)
            batch_size = X.size(0)
            input_seq = X.clone()
            preds = []

            for step in range(n_steps):
                pred = model(input_seq).squeeze(-1)  # (batch_size)
                preds.append(pred.cpu().numpy())

                # 将预测值加入输入序列
                pred_input = pred.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
                input_seq = torch.cat((input_seq[:, :, 1:], pred_input), dim=2)

            preds = np.array(preds).T  # 转置，使其形状为(batch_size, n_steps)
            y_pred.extend(preds.tolist())
            y_true.extend(Y.cpu().numpy().tolist())

    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    if val_mode:
        valMSE = mean_squared_error(y_true=y_true, y_pred=y_pred)
        return valMSE
    else:
        r2Score = r2_score(y_true=y_true, y_pred=y_pred)
        meanSquaredError = mean_squared_error(y_true=y_true, y_pred=y_pred)
        meanAbsoluteError = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        print("r2Score: ", r2Score)
        print("meanSquaredError: ", meanSquaredError)
        print('RMSE: ', sqrt(meanSquaredError))
        print("meanAbsoluteError: ", meanAbsoluteError)

        # 可视化预测结果和真实结果
        plt.plot(range(len(y_true[:1000])), y_pred[:1000], color='red', linewidth=1.5, linestyle='-.', label='prediction')
        plt.plot(range(len(y_true[:1000])), y_true[:1000], color='blue', linewidth=1.5, linestyle='-', label='real')
        plt.legend(loc='best')
        plt.show()

        return y_pred, y_true
##用真实的值预测
# def CNNBiLstm_evaluate(model, loader, config, val_mode=False):
#     model.eval()

#     y = list()
#     y_pre = list()
#     for idx, (X, Y) in tqdm(enumerate(loader)):
#         # Conv1d接受的数据输入是(batch_size有, channel_size=1, seq_len有),故增加一个通道数，单序列通道数为1，第2维
#         X = X.unsqueeze(-2).to(config.device)
#         Y = Y.to(config.device)
        
#         y_pre += model(X).cpu().squeeze().tolist()
#         y += Y.cpu().squeeze().tolist()

#     if val_mode:
#         valmeanSquaredError = mean_squared_error(y_true=y, y_pred=y_pre)
#         return valmeanSquaredError
#     else:
#         r2Score = r2_score(y_true=y, y_pred=y_pre)
#         meanSquaredError = mean_squared_error(y_true=y, y_pred=y_pre)
#         meanAbsoluteError = mean_absolute_error(y_true=y, y_pred=y_pre)
#         print("r2Score: ", r2Score)
#         print("meanSquaredError: ", meanSquaredError)
#         print('RMSE: ',sqrt(meanSquaredError))
#         print("meanAbsoluteError: ", meanAbsoluteError)

#         # 画出实际结果和预测的结果
#         import matplotlib.pyplot as plt
#         plt.plot(range(len(y[:1000])),y_pre[:1000],color = 'red',linewidth = 1.5,linestyle = '-.',label='prediction')
#         plt.plot(range(len(y[:1000])),y[:1000],color = 'blue',linewidth = 1.5,linestyle = '-', label='real')
#         plt.legend(loc='best')

#         return y_pre,y
        
#def CNNBiLstm_evaluate(model, loader, config, val_mode=False):
#     model.eval()

#     y = list()
#     y_pre = list()
#     for idx, (X, Y) in tqdm(enumerate(loader)):
#         # Conv1d接受的数据输入是(batch_size有, channel_size=1, seq_len有),故增加一个通道数，单序列通道数为1，第2维
#         X = X.unsqueeze(-2).to(config.device)
#         Y = Y.to(config.device)
        
#         y_pre += model(X).cpu().squeeze().tolist()
#         y += Y.cpu().squeeze().tolist()

#     if val_mode:
#         valmeanSquaredError = mean_squared_error(y_true=y, y_pred=y_pre)
#         return valmeanSquaredError
#     else:
#         r2Score = r2_score(y_true=y, y_pred=y_pre)
#         meanSquaredError = mean_squared_error(y_true=y, y_pred=y_pre)
#         meanAbsoluteError = mean_absolute_error(y_true=y, y_pred=y_pre)
#         print("r2Score: ", r2Score)
#         print("meanSquaredError: ", meanSquaredError)
#         print('RMSE: ',sqrt(meanSquaredError))
#         print("meanAbsoluteError: ", meanAbsoluteError)

#         # 画出实际结果和预测的结果
#         import matplotlib.pyplot as plt

# # 创建残差
#         residuals = [a - b for a,b in zip(y, y_pre)]
#         mse_per_point = [(s - p) ** 2 for s, p in zip(y, y_pre)]

# #创建一个图形，并且分为三行一列
#         fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(17, 8), sharex=True)

# #绘制第一行（预测值）
#         ax1.plot(range(len(y[:1000])), y_pre[:1000], color='red', linewidth=1.5, linestyle='-', label='Prediction')
#         ax1.plot(range(len(y[:1000])), y[:1000], color='blue', linewidth=1.5, linestyle='-', label='Real')

#         ax1.set_ylabel('Prediction')
#         ax1.legend(loc='best')
#         ax1.set_title('LSTM Predictions vs Real Data')

# # 绘制第二行（实际值）
#         ax2.plot(range(len(y[:1000])), y[:1000], color='blue', linewidth=1.5, linestyle='-', label='Real')
#         ax2.plot(range(len(residuals[:1000])), residuals[:1000], color='green', linestyle='-', label='Residual')

#         ax2.set_ylabel('Real')
#         ax2.legend(loc='best')

# # 绘制第三行（残差）
#         ax3.plot(range(len(residuals[:1000])), residuals[:1000], color='green', linestyle='-', label='Residual')
#         ax3.set_xlabel('Time')
#         ax3.set_ylabel('Residual')
#         ax3.legend(loc='best')

#         ax4.plot(range(len(y[:1000])), mse_per_point[:1000], color='orange', linestyle='-', label='mse')
#         ax4.set_xlabel('time')
#         ax4.set_ylabel('mse')
#         ax4.legend(loc='best')

# # 调整布局，防止重叠
#         plt.tight_layout()
#         plt.show()
#         print(residuals[:5])
#         print(y[:5])
#         print(y_pre[:5])


#         return y_pre, y, r2Score, residuals

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
##用预测的值预测
def CNNBiLstmtrain(model, trainloader, valloader, criterion, optimizer, config):
    logfile = open("./log/trainLog.txt", mode='a+', encoding='utf-8')
    n_steps = config.n_steps  # 预测的时间步数
    for epoch in range(config.epoch_size):
        model.train()
        for idx, (X, Y) in enumerate(trainloader):
            X = X.unsqueeze(-2).to(config.device)  # (batch_size, 1, lookback)
            Y = Y.to(config.device)  # (batch_size, n_steps)
            batch_size = X.size(0)
            lookback = X.size(2)

            # 初始化输入序列
            input_seq = X.clone()
            total_loss = 0

            # 在每个时间步进行预测
            for step in range(n_steps):
                # 模型预测
                pred = model(input_seq).squeeze(-1)  # (batch_size)
                true = Y[:, step]  # 当前时间步的真实值

                # 计算损失
                loss = criterion(pred, true)
                total_loss += loss

                # 将预测值加入输入序列，移除最早的数据点
                pred_input = pred.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
                input_seq = torch.cat((input_seq[:, :, 1:], pred_input), dim=2)  # 更新输入序列

            # 反向传播和参数更新
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                avg_loss = total_loss.item() / n_steps
                print(f"Epoch: {epoch} Batch: {idx} | Loss: {avg_loss}")

        # 一个epoch结束，进行一次验证
        valMSE = CNNBiLstm_evaluate(model, loader=valloader, config=config, val_mode=True)
        print(f"Epoch: {epoch} Validation Loss: {valMSE}")
        logfile.write(f"Epoch: {epoch} Validation Loss: {valMSE}\n")

    logfile.write("\n")
    logfile.close()
    return model
##用真实的值预测
# def CNNBiLstmtrain(model, trainloader,valloader, criterion, optimizer, config):
#     logfile = open("./log/trainLog.txt",mode='a+',encoding='utf-8')
#     for epoch in range(config.epoch_size):
#         model.train()
#         for idx, (X, Y) in enumerate(trainloader):
#             # Conv1d接受的数据输入是(batch_size有, channel_size=1, seq_len有),故增加一个通道数，单序列通道数为1，第2维
#             X = X.unsqueeze(-2).to(config.device)
#             Y = Y.to(config.device)
            
#             predict = model(X)
#             loss = criterion(predict, Y)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if idx % 10 == 0:
#                 print(f"Epoch: {epoch} batch: {idx} | loss: {loss}")
                
        
#         # 一个epoch结束,进行一次验证,并保存相关记录
#         valMSE = CNNBiLstm_evaluate(model,loader = valloader,config=config,val_mode=True)
#         print(f"Epoch: {epoch} valLoss: {valMSE}")
#         logfile.write("Epoch: {0} valLoss: {1} \n".format(epoch,valMSE))

#     logfile.write("\n")
#     logfile.close()
#     return model
