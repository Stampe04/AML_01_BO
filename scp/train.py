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

    # To hold accuracy and loss during training and testing
    train_accs = []
    test_accs = []
    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        epoch_acc = 0
        epoch_loss = 0
        num_batches = 0

        for inputs, targets in tqdm(train_dataloader):
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)

            logits = model(inputs)
            loss = model.criterion(logits, targets)
            loss.backward()

            model.optim.step()
            model.optim.zero_grad()

            # Keep track of training accuracy and loss
            epoch_acc += (torch.argmax(logits, dim=1) == targets).sum().item()
            epoch_loss += loss.item()
            num_batches += 1

        train_accs.append(epoch_acc / len(train_dataloader.dataset))
        train_losses.append(epoch_loss / num_batches)

        # If val_dataloader, evaluate after each epoch
        if val_dataloader is not None:
            # Turn off dropout for testing
            model.eval()
            acc = utils.eval_model(model, val_dataloader, device=device)
            test_accs.append(acc)
            val_loss = utils.eval_loss(model, val_dataloader, device=device)
            val_losses.append(val_loss)
            
            # turn on dropout after being done
            model.train()

    return train_accs, test_accs, train_losses, val_losses