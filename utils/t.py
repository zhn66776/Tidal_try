def CNNBiLstmtrain(model, trainloader, valloader, criterion, optimizer, config):
    for epoch in range(config.epoch_size):
        model.train()
        for idx, (X, Y) in enumerate(trainloader):
            X = X.to(config.device)
            Y = Y.to(config.device)

            predict = model(X)
            loss = criterion(predict, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = CNNBiLstm_evaluate_validation(model, valloader, criterion, config)
        print(f"Epoch {epoch + 1}/{config.epoch_size}, Validation Loss: {val_loss:.4f}")
    return model


def CNNBiLstm_evaluate_validation(model, valloader, criterion, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, Y in valloader:
            X = X.to(config.device)
            Y = Y.to(config.device)
            predict = model(X)
            loss = criterion(predict, Y)
            total_loss += loss.item()
    return total_loss / len(valloader)
def CNNBiLstm_evaluate(model, initial_sequence, test_Y, utide_test, config):
    model.eval()
    y_pred = []
    current_sequence = initial_sequence.squeeze().tolist()

    for i in range(len(test_Y)):
        input_seq = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).to(config.device)
        with torch.no_grad():
            predicted_value = model(input_seq).cpu().item()
        y_pred.append(predicted_value)
        current_sequence.pop(0)
        current_sequence.append([predicted_value, utide_test[len(current_sequence)]])  # 添加真实utide值

    return y_pred, test_Y
