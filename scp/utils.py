import torch


def eval_model(model, test_dataloader, device=None):
    was_training = model.training
    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = None

    total_acc = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)

            logits = model(inputs)
            total_acc += (torch.argmax(logits, dim=1) == targets).sum().item()

    total_acc = total_acc / len(test_dataloader.dataset)

    if was_training:
        model.train()

    return total_acc


def get_dim_before_first_linear(features, in_width_height, in_channels, device=None, brain=False):
    if device is None:
        try:
            device = next(features.parameters()).device
        except StopIteration:
            device = None

    dummy_input = torch.zeros(1, in_channels, in_width_height, in_width_height)
    if device is not None:
        dummy_input = dummy_input.to(device)

    with torch.no_grad():
        out = features(dummy_input)

    return out.shape[1]
