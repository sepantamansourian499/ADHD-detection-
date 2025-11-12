# preprocessing.py
import os, glob, json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert

def _read_csv_auto(path: str) -> np.ndarray:
    """CSV → ndarray float32, حذف ستون/سطر تمام-NaN، و تبدیل به [C,T]."""
    df = pd.read_csv(path, sep=None, engine="python")
    num = df.apply(pd.to_numeric, errors="coerce")
    num = num.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if num.empty:
        raise ValueError(f"No numeric data in {path}")
    arr = num.values.astype(np.float32)
    # اگر ردیف بیشتر از ستون بود، یعنی احتمالاً [T,C] → transpose
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr  # [C, T]

def _butter_band(sig: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(N=4, Wn=[lo/nyq, hi/nyq], btype="band")
    return filtfilt(b, a, sig, axis=-1)

def _band_env(sig: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    bp = _butter_band(sig, fs, lo, hi)
    analytic = hilbert(bp, axis=-1)
    return np.abs(analytic)

def _to_rgb(segment_ct: np.ndarray, fs: float) -> np.ndarray:
    """یک پنجره [C,win] → تصویر RGB به صورت [C,win,3] (theta/alpha/beta-gamma)."""
    R = _band_env(segment_ct, fs, 4.0, 8.0)    # theta
    G = _band_env(segment_ct, fs, 8.0, 12.0)   # alpha
    B = _band_env(segment_ct, fs, 12.0, 40.0)  # beta/gamma

    def mm(x):
        mn = x.min(axis=-1, keepdims=True)
        mx = x.max(axis=-1, keepdims=True)
        d  = np.clip(mx - mn, 1e-8, None)
        return (x - mn) / d

    rgb = np.stack([mm(R), mm(G), mm(B)], axis=-1)  # [C,win,3]
    return rgb.astype(np.float32)

def segment_no_overlap(x_ct: np.ndarray, win: int) -> List[np.ndarray]:
    """پنجره‌بندی بدون overlap → قدم = win."""
    C, T = x_ct.shape
    out = []
    for s in range(0, T - win + 1, win):
        out.append(x_ct[:, s:s+win])
    return out

def build_windows_to_disk(base_dir: str, out_dir: str,
                          fs: float = 128.0, win: int = 512) -> Dict:
    """
    از پوشه‌های ADHD/ و Normal/ می‌خواند، بدون overlap پنجره‌بندی می‌کند،
    هر پنجره را به RGB تبدیل می‌کند و به صورت .npy ذخیره می‌کند.
    """
    out = Path(out_dir)
    (out / "processed").mkdir(parents=True, exist_ok=True)

    class_to_idx = {"ADHD": 0, "Normal": 1}
    items = []

    for cls in ["ADHD", "Normal"]:
        y = class_to_idx[cls]
        pattern = os.path.join(base_dir, cls, "*.csv")
        for csv_path in glob.glob(pattern):
            gid = Path(csv_path).stem  # شناسه فایل برای جلوگیری از leakage
            x_ct = _read_csv_auto(csv_path)  # [C,T] (فرض fs=128 است)

            # پنجره‌ها
            wins = segment_no_overlap(x_ct, win)
            if not wins:
                continue

            for i, w in enumerate(wins):
                rgb = _to_rgb(w, fs=fs)  # [C,win,3]
                save_path = out / "processed" / f"{gid}__w{i:04d}__y{y}.npy"
                np.save(save_path.as_posix(), rgb)
                items.append({"x": save_path.as_posix(), "y": int(y), "g": gid})

    if not items:
        raise RuntimeError("No windows produced. Adjust 'win' or check CSVs.")

    manifest = {"class_to_idx": class_to_idx,
                "fs": fs, "window": win, "overlap": 0.0,
                "items": items}
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    with open(out / "manifests" / "windows.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
