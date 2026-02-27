import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import dataloader
import model
import train
import utils
from BO import skopt_BO

def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG16 on Imagenette.")
    parser.add_argument("--dataset", default="imagenette")
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--imagenette-resize", type=int, default=256)
    parser.add_argument("--imagenette-crop", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--acquisition-function", default="PI", help="Acquisition function for BO (e.g., EI, PI, LCB)")
    parser.add_argument("--plot-path", default="test_accuracy_runs.png")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    BO_model = skopt_BO(model=model.VGG16(num_classes=args.num_classes, in_channels=3), 
                        min_kernel_number=16, max_kernel_number=64, min_dropout_rate=0.0, 
                        max_dropout_rate=0.8, acquisition_function=args.acquisition_function)
    
    # Update plot paths to include acquisition function
    acq_func = args.acquisition_function
    base_plot_name = args.plot_path.replace('.png', '')
    args.plot_path = f"{base_plot_name}_{acq_func}.png"

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

    test_accs = []
    train_histories = []
    val_histories = []

    for run_idx in range(args.runs):

        kernel_number, dropout = BO_model.suggest()

        cnn_model = model.VGG16(
            num_classes=args.num_classes,
            in_channels=in_channels,
            num_kernels=kernel_number,
            dropout_rate=dropout,
            dataset=test_set,
        )

        train_accs, val_accs, train_losses, val_losses = train.train_model(
            cnn_model,
            train_dataloader,
            epochs=args.epochs,
            val_dataloader=validation_dataloader,
            device=device,
        )

        # Use validation accuracy for Bayesian Optimization
        val_acc = val_accs[-1] if val_accs else 0.0
        
        # Evaluate on test set only for reporting
        test_acc = utils.eval_model(cnn_model, test_dataloader, device=device)
        test_accs.append(test_acc)
        train_histories.append(train_accs)
        val_histories.append(val_accs)
        print(f"Run {run_idx + 1}/{args.runs} - Val accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}")
        print(f"Run {run_idx + 1} BO params - kernel_number: {kernel_number}, dropout_rate: {dropout:.4f}")


        # Plot and save loss for this run
        if len(train_losses) > 0:
            loss_plot_path = f"loss_run_{run_idx + 1}_{acq_func}.png"
            plt.figure(figsize=(7, 4))
            plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label="Train loss")
            if len(val_losses) > 0:
                plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', label="Val loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Loss - Run {run_idx + 1}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"Saved loss plot to {loss_plot_path}")

        # Update BO with validation accuracy
        BO_model.update(kernel_number, dropout, val_acc)

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
    
    print(f"using acquisition function: {acq_func}")
    # Add accuracy values as text labels above each point
    for run_id, acc in zip(run_ids, test_accs):
        plt.text(run_id, acc, f'{acc:.4f}', ha='center', va='bottom', fontsize=8)
    
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
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(args.plot_path)
    print(f"Saved plot to {args.plot_path}")

    # Create boxplot
    boxplot_path = f"{base_plot_name}_boxplot_{acq_func}.png"
    plt.figure(figsize=(7, 4))
    bp = plt.boxplot([test_accs], vert=True, patch_artist=True, labels=['Test Accuracy'])
    bp['boxes'][0].set_facecolor('tab:blue')
    bp['boxes'][0].set_alpha(0.7)
    plt.axhline(mean_acc, color='tab:orange', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.4f}')
    plt.ylabel("Test accuracy")
    plt.title("Distribution of test accuracy across runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(boxplot_path)
    print(f"Saved boxplot to {boxplot_path}")

    return train_histories, val_histories, test_accs


if __name__ == "__main__":
    main(parse_args())
