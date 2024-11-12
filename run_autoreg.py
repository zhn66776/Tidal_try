import torch
from torch.utils.data import DataLoader
from config import Config

from utils.dataTools import mydataReader, custom_dataset
from utils.models import CNNBiLSTM
from utils.trainer import CNNBiLstm_evaluate, CNNBiLstmtrain

if __name__ == '__main__':
    config = Config()

    print("Data loading...")
    # 序列数据
    dataset = mydataReader("./dataProcessed/procan.csv")

    # 创建X/Y
    # 划分训练集和测试集，70% 作为训练集,10%作为验证集,20%作为测试集
    (train_X, train_Y), (val_X, val_Y), (test_X, test_Y) = dataset.split(lookback=config.lookback, trainSet_ratio=0.833, valSet_ratio=0.08)

    # 创建Pytorch使用的dataset
    trainSet = custom_dataset(train_X, train_Y)
    valSet = custom_dataset(val_X, val_Y)
    # 不再需要创建 testSet

    train_loader = DataLoader(trainSet, batch_size=config.batch_size,
                              shuffle=False, pin_memory=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(valSet, batch_size=config.batch_size,
                            shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    # 不再需要创建 test_loader

    print("Model loading...")
    model = CNNBiLSTM(hidden_size=12, num_layers=3).to(config.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    print(f"Configuration Parameters:")
    print(f" - Batch size: {config.batch_size}")
    print(f" - Lookback: {config.lookback}")
    print(f" - Epochs: {config.epoch_size}")
    print(f" - Learning rate: {config.learning_rate}")
    print(f" - Weight decay: {config.weight_decay}")

    print("Training...")
    model = CNNBiLstmtrain(model,
                           trainloader=train_loader,
                           valloader=val_loader,
                           criterion=criterion,
                           optimizer=optimizer,
                           config=config)

    print("Testing...")
    # 使用测试集的第一个输入序列作为初始序列
    initial_sequence = test_X[0]

    # 调用修改后的评估函数
    y_pred, y_true = CNNBiLstm_evaluate(model, initial_sequence, test_Y, config)

