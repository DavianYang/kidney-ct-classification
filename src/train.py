import random

import numpy as np

import torch
from torchvision.transforms import v2
from torch.utils.data import Subset, DataLoader

from sklearn.model_selection import train_test_split

from pathlib import Path
from src.dataset import KidneyDataset


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Kidney Stone CT Classification Training", add_help=add_help
    )

    # 1. Hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Max number of epochs")
    parser.add_argument(
        "--args.batch_size", type=int, default=64, help="Batch size for dataloaders"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--delta", type=float, default=1e-4, help="Min change for early stopping"
    )

    # 2. Paths & Saving
    parser.add_argument(
        "--model_name",
        type=str,
        default="my_model",
        help="Name prefix for saved weights",
    )
    parser.add_argument(
        "--topk", type=int, default=3, help="Number of best models to keep"
    )

    # 3. Hardware/Environment
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no_progress",
        action="store_false",
        dest="show_progress",
        help="Hide progress bar",
    )

    return parser


def main(args):
    torch.manual_seed(args.seed)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training {args.model_name} on {device}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
