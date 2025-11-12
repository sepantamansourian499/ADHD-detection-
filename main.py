
import os, random
os.environ["PYTHONHASHSEED"] = "123"                 # affects Python hashing (set early)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # CUDA determinism

import argparse, json
from pathlib import Path

import numpy as np
np.random.seed(123)  # will re-seed again with args.seed later

import torch

# -------- Repro helpers --------
def set_seed(seed: int = 123, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # older PyTorch versions

def seed_worker(worker_id: int):
    """Use with DataLoader(worker_init_fn=seed_worker)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_generator(seed: int = 123):
    g = torch.Generator()
    g.manual_seed(seed)
    return g
# --------------------------------

from preprocessing import build_windows_to_disk
from dataset import split_by_file_no_leak, make_loaders
from model import EEGSmallCNN
from train import train_eval


def parse_args():
    ap = argparse.ArgumentParser("EEG→RGB→CNN (overlap)")
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--out_dir",  type=str, default="outputs")
    ap.add_argument("--epochs",   type=int, default=20)
    ap.add_argument("--batch",    type=int, default=8)
    ap.add_argument("--fs",       type=float, default=128.0)
    ap.add_argument("--window",   type=int, default=512)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()

    # Re-seed everything with the user-provided seed
    set_seed(args.seed)
    gen = make_generator(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build RGB windows on disk (if not already present)
    manifest = build_windows_to_disk(
        args.base_dir, out_dir / "data", fs=args.fs, win=args.window
    )

    # 2) Split without file/group leakage; stratified train/val
    splits = split_by_file_no_leak(
        manifest["items"], test_ratio=0.3, val_ratio=0.2, seed=args.seed
    )
    (out_dir / "split_stats.json").write_text(json.dumps({
        "train": len(splits["train"]),
        "val":   len(splits["val"]),
        "test":  len(splits["test"]),
        "seed":  args.seed
    }, indent=2), encoding="utf-8")

    # 3) DataLoaders (seeded workers + generator)
    dl_tr, dl_va, dl_te = make_loaders(
        splits,
        batch=args.batch,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=gen
    )

    # 4) Model + train/eval
    model = EEGSmallCNN(num_classes=2)
    train_eval(
        model,
        {"train": dl_tr, "val": dl_va, "test": dl_te},
        out_dir=str(out_dir),
        epochs=args.epochs,
        lr=1e-3,
        device="cuda"
    )


if __name__ == "__main__":
    main()

