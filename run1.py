import torch
from torch.utils.data import DataLoader
from config import Config

from utils.dataTools import *
from utils.models import CNNBiLSTM
from utils.trainer import CNNBiLstm_evaluate,CNNBiLstmtrain
if __name__ == '__main__':
    config = Config()

    print("Data loading...")

    dataset = mydataReader("./dataProcessed/IrishNationalTideGaugeNetwork_Ballycotton Harbour2017.csv")

    (train_X, train_Y), (val_X, val_Y), (test_X, test_Y) = dataset.split(
        lookback=config.lookback, n_steps=config.n_steps, trainSet_ratio=0.7, valSet_ratio=0.1)


    trainSet = custom_dataset(train_X, train_Y)
    valSet = custom_dataset(val_X, val_Y)
    testSet = custom_dataset(test_X, test_Y)

    train_loader = DataLoader(trainSet, batch_size=config.batch_size,
                              shuffle=False, pin_memory=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(valSet, batch_size=config.batch_size,
                            shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    test_loader = DataLoader(testSet, batch_size=config.batch_size,
                             shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    print("Model loading...")
    model = CNNBiLSTM(hidden_size=12, num_layers=2).to(config.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.learning_rate, weight_decay=config.weight_decay)

    print("Training...")
    model = CNNBiLstmtrain(model,
                           trainloader=train_loader,
                           valloader=val_loader,
                           criterion=criterion,
                           optimizer=optimizer,
                           config=config)

    print("Testing...")
    CNNBiLstm_evaluate(model, test_loader, config)
