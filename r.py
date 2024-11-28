if __name__ == '__main__':
    from config import Config
    config = Config()

    dataset = mydataReader("./dataProcessed/procan.csv")
    (train_X, train_Y), (val_X, val_Y), (test_X, test_Y) = dataset.split(
        lookback=config.lookback, trainSet_ratio=0.833, valSet_ratio=0.08
    )
    test_utide = dataset.data_csv['utide'].values[-len(test_Y):]

    trainSet = custom_dataset(train_X, train_Y)
    valSet = custom_dataset(val_X, val_Y)

    train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(valSet, batch_size=config.batch_size, shuffle=False)

    model = CNNBiLSTM(hidden_size=12, num_layers=3).to(config.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    model = CNNBiLstmtrain(model, train_loader, val_loader, criterion, optimizer, config)

    initial_sequence = test_X[0]
    y_pred, y_true = CNNBiLstm_evaluate(model, initial_sequence, test_Y, test_utide, config)
