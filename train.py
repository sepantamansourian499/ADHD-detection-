# train.py
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def _acc(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def train_eval(model, loaders, out_dir="outputs", epochs=50, lr=1e-3, device="cuda"):
    out = Path(out_dir); (out / "figs").mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for ep in range(1, epochs+1):
        # --- train
        model.train()
        tl, ta, n = 0.0, 0.0, 0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            bs = xb.size(0)
            tl += loss.item()*bs; ta += _acc(logits, yb)*bs; n += bs
        tl, ta = tl/n, ta/n

        # --- val
        model.eval()
        vl, va, n = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                bs = xb.size(0)
                vl += loss.item()*bs; va += _acc(logits, yb)*bs; n += bs
        vl, va = vl/n, va/n

        hist["train_loss"].append(tl); hist["train_acc"].append(ta)
        hist["val_loss"].append(vl);   hist["val_acc"].append(va)
        print(f"Epoch {ep:03d} | train: loss={tl:.4f} acc={ta*100:.2f}% | val: loss={vl:.4f} acc={va*100:.2f}%")

    # curves
    np.savetxt(out/"train_loss.csv", np.array(hist["train_loss"]), delimiter=",")
    np.savetxt(out/"train_acc.csv",  np.array(hist["train_acc"]),  delimiter=",")
    np.savetxt(out/"val_loss.csv",   np.array(hist["val_loss"]),   delimiter=",")
    np.savetxt(out/"val_acc.csv",    np.array(hist["val_acc"]),    delimiter=",")
    plt.figure(); plt.plot(hist["train_acc"], label="train"); plt.plot(hist["val_acc"], label="val"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("acc"); plt.tight_layout(); plt.savefig(out/"figs"/"acc_curve.png", dpi=180); plt.close()
    plt.figure(); plt.plot(hist["train_loss"], label="train"); plt.plot(hist["val_loss"], label="val"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.tight_layout(); plt.savefig(out/"figs"/"loss_curve.png", dpi=180); plt.close()

    # --- test
    ys, ps, n, tlos = [], [], 0, 0.0
    model.eval()
    with torch.no_grad():
        for xb, yb in loaders["test"]:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            tlos += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
            ys.append(yb.cpu().numpy()); ps.append(logits.argmax(1).cpu().numpy())
            n += xb.size(0)
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    test_loss = tlos/n; test_acc = (ys==ps).mean()
    print(f"[Test] loss={test_loss:.4f} acc={test_acc*100:.2f}%")

    # report & confusion matrix
    rep = classification_report(ys, ps, target_names=["ADHD","Normal"], digits=4, zero_division=0)
    (out/"classification_report.txt").write_text(rep, encoding="utf-8")

    cm = confusion_matrix(ys, ps)
    plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix"); plt.colorbar()
    ticks = range(len(["ADHD","Normal"]))
    plt.xticks(ticks, ["ADHD","Normal"]); plt.yticks(ticks, ["ADHD","Normal"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center")
    plt.xlabel("Pred"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(out/"figs"/"confusion_matrix.png", dpi=180); plt.close()
