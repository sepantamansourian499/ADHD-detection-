
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

class RGBDataset(Dataset):
    def __init__(self, items: List[Dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        # .npy file shaped [C, win, 3]
        arr = np.load(rec["x"]).astype(np.float32)
        x = np.transpose(arr, (2, 0, 1))  # -> [3, C, win]
        y = int(rec["y"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def split_by_file_no_leak(items: List[Dict], test_ratio=0.3, val_ratio=0.2, seed=42,
                          val_ratio_is_global: bool = False) -> Dict[str, List[Dict]]:
    """
    1) Hold out test by group (no file leakage)
    2) Stratified train/val on remaining
    If val_ratio_is_global=True, val_ratio refers to the whole dataset.
    """
    X = np.array([it["x"] for it in items])
    y = np.array([it["y"] for it in items])
    g = np.array([it["g"] for it in items])

    # Test split by groups
    gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trval_idx, te_idx = next(gss.split(X, y, groups=g))

    X_trv, y_trv, g_trv = X[trval_idx], y[trval_idx], g[trval_idx]
    X_te,  y_te,  g_te  = X[te_idx],  y[te_idx],  g[te_idx]

    # Effective val size on the remaining pool
    if val_ratio_is_global:
        effective_val = val_ratio / max(1e-9, (1 - test_ratio))
    else:
        effective_val = val_ratio

    sss = StratifiedShuffleSplit(n_splits=1, test_size=effective_val, random_state=seed)
    tr_idx, va_idx = next(sss.split(X_trv, y_trv))

    def pack(Xi, yi, gi):
        return [{"x": p, "y": int(lbl), "g": gg} for p, lbl, gg in zip(Xi, yi, gi)]

    return {
        "train": pack(X_trv[tr_idx], y_trv[tr_idx], g_trv[tr_idx]),
        "val":   pack(X_trv[va_idx], y_trv[va_idx], g_trv[va_idx]),
        "test":  pack(X_te,          y_te,          g_te)
    }

def make_loaders(splits: Dict[str, List[Dict]], batch=8, num_workers=0,
                 worker_init_fn=None, generator=None, pin_memory=None):
    """
    Pass seed helpers from main:
        worker_init_fn=seed_worker, generator=gen
    """
    ds_tr = RGBDataset(splits["train"])
    ds_va = RGBDataset(splits["val"])
    ds_te = RGBDataset(splits["test"])

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dl_tr = DataLoader(
        ds_tr, batch_size=batch, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, worker_init_fn=worker_init_fn, generator=generator,
        persistent_workers=(num_workers > 0)
    )
    dl_va = DataLoader(
        ds_va, batch_size=batch, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, worker_init_fn=worker_init_fn, generator=generator,
        persistent_workers=(num_workers > 0)
    )
    dl_te = DataLoader(
        ds_te, batch_size=batch, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, worker_init_fn=worker_init_fn, generator=generator,
        persistent_workers=(num_workers > 0)
    )
    return dl_tr, dl_va, dl_te

