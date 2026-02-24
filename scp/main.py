import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import dataloader
from . import model
from . import train
from . import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG16 on Imagenette.")
    parser.add_argument("--dataset", default="imagenette")
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--imagenette-resize", type=int, default=256)
    parser.add_argument("--imagenette-crop", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--plot-path", default="test_accuracy_runs.png")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, validation_set, test_set = dataloader.get_dataset(
        args.dataset,
        validation_size=args.validation_size,
        imagenette_resize_size=args.imagenette_resize,
        imagenette_crop_size=args.imagenette_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: dataloader.collate_fn(batch, device=device),
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: dataloader.collate_fn(batch, device=device),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: dataloader.collate_fn(batch, device=device),
    )

    in_channels = next(iter(train_dataloader))[0].shape[1]
    in_width_height = next(iter(train_dataloader))[0].shape[-1]

    temp_model = model.VGG16(num_classes=args.num_classes, in_channels=in_channels)
    features_fore_linear = utils.get_dim_before_first_linear(
        temp_model.features, in_width_height, in_channels
    )

    test_accs = []
    train_histories = []
    val_histories = []

    for run_idx in range(args.runs):
        seed_everything(args.seed + run_idx)

        cnn_model = model.VGG16(
            num_classes=args.num_classes,
            in_channels=in_channels,
            features_fore_linear=features_fore_linear,
            dataset=test_set,
        )

        train_accs, val_accs = train.train_model(
            cnn_model,
            train_dataloader,
            epochs=args.epochs,
            val_dataloader=validation_dataloader,
            device=device,
        )

        test_acc = utils.eval_model(cnn_model, test_dataloader, device=device)
        test_accs.append(test_acc)
        train_histories.append(train_accs)
        val_histories.append(val_accs)
        print(f"Run {run_idx + 1}/{args.runs} test accuracy: {test_acc}")

    test_accs = np.array(test_accs, dtype=np.float32)
    mean_acc = float(test_accs.mean())
    std_acc = float(test_accs.std(ddof=1)) if len(test_accs) > 1 else 0.0
    z = float(torch.distributions.Normal(0, 1).icdf(torch.tensor(0.5 + args.ci / 2)))
    half_width = z * std_acc / (len(test_accs) ** 0.5) if len(test_accs) > 1 else 0.0

    print(f"Mean test accuracy: {mean_acc}")
    print(f"Std test accuracy: {std_acc}")
    print(f"{int(args.ci * 100)}% CI: [{mean_acc - half_width}, {mean_acc + half_width}]")

    run_ids = np.arange(1, len(test_accs) + 1)
    plt.figure(figsize=(7, 4))
    plt.scatter(run_ids, test_accs, label="Run accuracy", zorder=3)
    plt.hlines(mean_acc, run_ids.min(), run_ids.max(), colors="tab:orange", label="Mean")
    plt.fill_between(
        [run_ids.min(), run_ids.max()],
        mean_acc - std_acc,
        mean_acc + std_acc,
        alpha=0.2,
        label="Mean +/- std",
        color="tab:orange",
    )
    plt.xlabel("Run")
    plt.ylabel("Test accuracy")
    plt.title("Test accuracy across runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_path)
    print(f"Saved plot to {args.plot_path}")

    return train_histories, val_histories, test_accs


if __name__ == "__main__":
    main(parse_args())
