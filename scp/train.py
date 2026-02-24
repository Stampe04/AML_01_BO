import torch

from tqdm import tqdm

import utils


def train_model(model, train_dataloader, epochs=1, val_dataloader=None, device=None):

    # Call .train() on model to turn on dropout
    model.train()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = None

    # To hold accuracy during training and testing
    train_accs = []
    test_accs = []

    for epoch in range(epochs):

        epoch_acc = 0

        for inputs, targets in tqdm(train_dataloader):
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)

            logits = model(inputs)
            loss = model.criterion(logits, targets)
            loss.backward()

            model.optim.step()
            model.optim.zero_grad()

            # Keep track of training accuracy
            epoch_acc += (torch.argmax(logits, dim=1) == targets).sum().item()

        train_accs.append(epoch_acc / len(train_dataloader.dataset))

        # If val_dataloader, evaluate after each epoch
        if val_dataloader is not None:
            # Turn off dropout for testing
            model.eval()
            acc = utils.eval_model(model, val_dataloader, device=device)
            test_accs.append(acc)
            print(f"Epoch {epoch} validation accuracy: {acc}")
            # turn on dropout after being done
            model.train()

    return train_accs, test_accs