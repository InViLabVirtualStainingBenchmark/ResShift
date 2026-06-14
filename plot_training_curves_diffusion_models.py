"""
plot_training_curves_diffusion_models.py
========================
Training curve plotter for diffusion virtual staining models.

Supported log formats (auto-detected):

    SinSR native format:
        Train: 001620/024400: loss:2.01e-02 mse:4.34e-02 mse_gt:3.18e-03 lr:7.99e-06
        Validation Metric: PSNR=14.41, LPIPS=0.7878

    ResShift native format (comma separator, timestep-specific losses):
        Train: 000001/000050, Loss/MSE: t(1):0.0e+00/0.0e+00, t(8):8.2e-01/8.2e-01, lr:5.00e-05

    Palette / Python-logging format:
        26-04-29 10:39:55.647 - INFO: epoch: 1
        26-04-29 10:39:55.647 - INFO: train/mse_loss: 1.585
        26-04-29 10:39:55.647 - INFO: val/mae: 0.740

    SR3 / junyanz-based format:
        (epoch: N, iters: M, time: T, data: D) KEY: V KEY: V ...
        End of epoch N / total   Time Taken: S sec
        learning rate = V

SinSR/ResShift x-axis uses raw iteration numbers.
Palette/SR3 x-axis uses epoch numbers.

Usage
-----
    python plot_training_curves_diffusion_models.py \\
        --logs  path/to/run.out \\
        --labels  ModelName-Stain \\
        --name  <stain> \\
        --out-dir  results/training_curves/<model>

    # Example single run — SinSR on BCI:
    python plot_training_curves_diffusion_models.py \\
        --logs  outputs/sinsr_bci.out \\
        --labels  SinSR-BCI \\
        --name  bci \\
        --out-dir  results/training_curves/sinsr

Outputs:
    <name>_training_curves.png
    <name>_training_summary.csv

Auto-shown panels (only when data is present):
    Diffusion loss          total loss, l_pix, t{N}_loss (ResShift timesteps), ...
    Distillation losses     mse / mse_gt  (SinSR distillation)
    Val LPIPS               primary convergence metric  (↓ better)
    Val PSNR / SSIM         image quality on validation set  (↑ better)
    Other validation        any remaining val_* keys
    Other losses            any remaining training keys
    Learning rate           LR schedule
    Epoch wall-clock time   training speed

Options
-------
    --smooth N       Moving-average window for training curves. Default: 5.
    --dpi N          Output image DPI. Default: 150.
    --last-n N       Final N epochs for summary averages. Default: 10.
                     Use 0 to average over all epochs.
"""

import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# junyanz / SR3 / Palette
_JITER_RE  = re.compile(r"\(epoch:\s*(\d+),\s*iters:\s*\d+.*?\)")
_JKV_RE    = re.compile(r"(\w+):\s*([\d.e+\-]+)")
_JEPOCH_RE = re.compile(r"End of epoch\s+(\d+)\s*/\s*\d+.*?Time Taken:\s*(\d+)")
_JLR_RE    = re.compile(r"learning rate\s*=\s*([\d.e+\-]+)")
_JHEADER   = {"epoch", "iters", "time", "data"}

# SinSR native (colon after total iters)
_STRAIN_RE = re.compile(r"Train:\s+(\d+)/\d+:\s+(.+)")
_SKVE_RE   = re.compile(r"(\w+):([\d.e+\-]+)")
_SVAL_RE   = re.compile(r"Validation Metric:\s+(.+)")
_SVKV_RE   = re.compile(r"(\w+)=(\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)")

# ResShift native (comma after total iters, timestep-specific Loss/MSE)
_RSHIFT_TRAIN_RE = re.compile(r"Train:\s+(\d+)/\d+,\s+Loss/MSE:\s+(.+)")
_RSHIFT_TS_RE    = re.compile(r"t\((\d+)\):([\d.e+\-]+)/([\d.e+\-]+)")
_RSHIFT_LR_RE    = re.compile(r"\blr:([\d.e+\-]+)")

# Palette / Python-logging format
_PAL_INFO_RE = re.compile(r"-\s+INFO:\s+([\w/]+):\s+([\d.e+\-]+)")
_PAL_TIME_RE = re.compile(r"^\d{2,4}-\d{2}-\d{2}\s+(\d{2}:\d{2}:\d{2})")

# Keys that are dataset/meta — excluded from loss groups and key iteration
_META_KEYS = frozenset(
    {"epochs", "x_label", "time_epochs", "time_sec", "lr_epochs", "lr",
     "_val_raw", "_x_is_iter"}
)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(path: str) -> str:
    """Return 'sinsr', 'resshift', 'palette', or 'junyanz'."""
    with open(path, errors="replace") as fh:
        for i, line in enumerate(fh):
            if i > 2000:
                break
            stripped = line.strip()
            if _RSHIFT_TRAIN_RE.match(stripped):
                return "resshift"
            if _STRAIN_RE.match(stripped):
                return "sinsr"
            if _JITER_RE.search(line):
                return "junyanz"
            if _PAL_INFO_RE.search(line):
                return "palette"
    return "junyanz"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_sinsr_log(path: str) -> dict:
    """
    Parse a SinSR native training log.

    X-axis uses raw iteration numbers.
    Validation metrics from 'Validation Metric: PSNR=V, LPIPS=V' lines
    are aligned to the nearest preceding logged iteration (NaN elsewhere).
    """
    train_buf: list[tuple[int, dict]] = []
    val_buf:   list[tuple[int, dict]] = []
    last_iter = 0

    with open(path, errors="replace") as fh:
        for line in fh:
            m = _STRAIN_RE.match(line.strip())
            if m:
                cur       = int(m.group(1))
                last_iter = cur
                kvs = {k: float(v) for k, v in _SKVE_RE.findall(m.group(2))}
                train_buf.append((cur, kvs))
                continue
            m = _SVAL_RE.search(line)
            if m:
                kvs = {k.lower(): float(v) for k, v in _SVKV_RE.findall(m.group(1))}
                if kvs:
                    val_buf.append((last_iter, kvs))

    if not train_buf:
        raise ValueError(f"No SinSR training lines found in {path}.")

    iter_train: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    lr_per_iter: dict[int, float] = {}

    for cur, kvs in train_buf:
        for k, v in kvs.items():
            if k == "lr":
                lr_per_iter[cur] = v
            else:
                iter_train[cur][k].append(v)

    iter_val: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for iter_at, kvs in val_buf:
        for k, v in kvs.items():
            iter_val[iter_at][k].append(v)

    iters          = sorted(iter_train.keys())
    all_train_keys = sorted({k for ep in iter_train.values() for k in ep})
    all_val_keys   = sorted({k for ep in iter_val.values()   for k in ep})

    out: dict = {
        "epochs":     np.array(iters, dtype=int),
        "x_label":    "Iteration",
        "_x_is_iter": True,
    }

    for key in all_train_keys:
        out[key] = np.array(
            [np.mean(iter_train[it][key]) if iter_train[it][key] else np.nan
             for it in iters],
            dtype=float,
        )

    for vk in all_val_keys:
        out[f"val_{vk}"] = np.array(
            [np.mean(iter_val[it].get(vk, [])) if iter_val[it].get(vk) else np.nan
             for it in iters],
            dtype=float,
        )

    it_lr = sorted(lr_per_iter)
    out["lr_epochs"] = np.array(it_lr, dtype=int)
    out["lr"]        = np.array([lr_per_iter[i] for i in it_lr], dtype=float)

    out["time_epochs"] = np.array([], dtype=int)
    out["time_sec"]    = np.array([], dtype=float)

    val_raw: dict = {}
    for vk in all_val_keys:
        pairs = sorted([(it, kvs[vk]) for it, kvs in val_buf if vk in kvs])
        if pairs:
            ri, rv = zip(*pairs)
            val_raw[vk] = (np.array(ri, dtype=int), np.array(rv, dtype=float))
    out["_val_raw"]    = val_raw

    return out


def parse_resshift_log(path: str) -> dict:
    """
    Parse a ResShift training log.

    Format: Train: ITER/TOTAL, Loss/MSE: t(N):loss/mse, ..., lr:V
    X-axis uses raw iteration numbers.
    Each timestep produces t{N}_loss and t{N}_mse keys.
    """
    train_buf: list[tuple[int, dict]] = []
    val_buf:   list[tuple[int, dict]] = []
    last_iter = 0

    with open(path, errors="replace") as fh:
        for line in fh:
            m = _RSHIFT_TRAIN_RE.match(line.strip())
            if m:
                cur       = int(m.group(1))
                last_iter = cur
                rest = m.group(2)
                kvs: dict[str, float] = {}
                for ts, loss_str, mse_str in _RSHIFT_TS_RE.findall(rest):
                    kvs[f"t{ts}_loss"] = float(loss_str)
                    kvs[f"t{ts}_mse"]  = float(mse_str)
                lr_m = _RSHIFT_LR_RE.search(rest)
                if lr_m:
                    kvs["lr"] = float(lr_m.group(1))
                train_buf.append((cur, kvs))
                continue
            m = _SVAL_RE.search(line)
            if m:
                kvs = {k.lower(): float(v) for k, v in _SVKV_RE.findall(m.group(1))}
                if kvs:
                    val_buf.append((last_iter, kvs))

    if not train_buf:
        raise ValueError(f"No ResShift training lines found in {path}.")

    iter_train: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    lr_per_iter: dict[int, float] = {}

    for cur, kvs in train_buf:
        for k, v in kvs.items():
            if k == "lr":
                lr_per_iter[cur] = v
            else:
                iter_train[cur][k].append(v)

    iter_val: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for iter_at, kvs in val_buf:
        for k, v in kvs.items():
            iter_val[iter_at][k].append(v)

    iters          = sorted(iter_train.keys())
    all_train_keys = sorted({k for ep in iter_train.values() for k in ep})
    all_val_keys   = sorted({k for ep in iter_val.values()   for k in ep})

    out: dict = {
        "epochs":     np.array(iters, dtype=int),
        "x_label":    "Iteration",
        "_x_is_iter": True,
    }
    for key in all_train_keys:
        out[key] = np.array(
            [np.mean(iter_train[it][key]) if iter_train[it][key] else np.nan
             for it in iters],
            dtype=float,
        )
    for vk in all_val_keys:
        out[f"val_{vk}"] = np.array(
            [np.mean(iter_val[it].get(vk, [])) if iter_val[it].get(vk) else np.nan
             for it in iters],
            dtype=float,
        )

    it_lr = sorted(lr_per_iter)
    out["lr_epochs"] = np.array(it_lr, dtype=int)
    out["lr"]        = np.array([lr_per_iter[i] for i in it_lr], dtype=float)
    out["time_epochs"] = np.array([], dtype=int)
    out["time_sec"]    = np.array([], dtype=float)

    val_raw: dict = {}
    for vk in all_val_keys:
        pairs = sorted([(it, kvs[vk]) for it, kvs in val_buf if vk in kvs])
        if pairs:
            ri, rv = zip(*pairs)
            val_raw[vk] = (np.array(ri, dtype=int), np.array(rv, dtype=float))
    out["_val_raw"]    = val_raw

    return out


def parse_palette_log(path: str) -> dict:
    """
    Parse a Palette Python-logging format training log.

    Expected lines (timestamp is optional):
        [TIMESTAMP] - INFO: epoch: N
        [TIMESTAMP] - INFO: train/KEY: VALUE
        [TIMESTAMP] - INFO: val/KEY: VALUE
        [TIMESTAMP] - INFO: lr: VALUE

    Treats epoch: N as marking the start of that epoch's block.
    Wall-clock time per epoch is computed from consecutive epoch timestamps.
    """
    epoch_train: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    epoch_val:   dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    epoch_time:  dict[int, float] = {}
    epoch_lr:    dict[int, float] = {}
    epoch_start_sec: dict[int, float] = {}

    current_epoch = 1

    with open(path, errors="replace") as fh:
        for line in fh:
            ts_m = _PAL_TIME_RE.match(line)
            cur_sec = None
            if ts_m:
                try:
                    h, mn, s = ts_m.group(1).split(":")
                    cur_sec = int(h) * 3600 + int(mn) * 60 + float(s)
                except (ValueError, IndexError):
                    pass

            info_m = _PAL_INFO_RE.search(line)
            if not info_m:
                continue

            key_full = info_m.group(1)
            val_str  = info_m.group(2)

            if key_full == "epoch":
                try:
                    current_epoch = int(float(val_str))
                    if cur_sec is not None and current_epoch not in epoch_start_sec:
                        epoch_start_sec[current_epoch] = cur_sec
                except ValueError:
                    pass
            elif key_full.startswith("train/"):
                k = key_full[6:]
                try:
                    epoch_train[current_epoch][k].append(float(val_str))
                except ValueError:
                    pass
            elif key_full.startswith("val/"):
                k = key_full[4:]
                try:
                    epoch_val[current_epoch][k].append(float(val_str))
                except ValueError:
                    pass
            elif key_full in ("lr", "learning_rate"):
                try:
                    epoch_lr[current_epoch] = float(val_str)
                except ValueError:
                    pass

    # Derive per-epoch wall-clock duration from consecutive epoch start times
    sorted_ep_ts = sorted(epoch_start_sec)
    for i in range(len(sorted_ep_ts) - 1):
        e1, e2 = sorted_ep_ts[i], sorted_ep_ts[i + 1]
        dt = epoch_start_sec[e2] - epoch_start_sec[e1]
        if dt < 0:
            dt += 86400  # midnight wrap
        epoch_time[e1] = dt

    epochs = sorted(set(epoch_train.keys()) | set(epoch_val.keys()))
    if not epochs:
        raise ValueError(f"No Palette training lines found in {path}.")

    all_train_keys = sorted({k for ep in epoch_train.values() for k in ep})
    all_val_keys   = sorted({k for ep in epoch_val.values()   for k in ep})

    out: dict = {
        "epochs":     np.array(epochs, dtype=int),
        "x_label":    "Epoch",
        "_x_is_iter": False,
    }
    for key in all_train_keys:
        out[key] = np.array(
            [np.mean(epoch_train[ep][key]) if epoch_train[ep][key] else np.nan
             for ep in epochs],
            dtype=float,
        )
    for vk in all_val_keys:
        out[f"val_{vk}"] = np.array(
            [np.mean(epoch_val[ep].get(vk, [])) if epoch_val[ep].get(vk) else np.nan
             for ep in epochs],
            dtype=float,
        )

    ep_t = sorted(epoch_time)
    out["time_epochs"] = np.array(ep_t, dtype=int)
    out["time_sec"]    = np.array([epoch_time[e] for e in ep_t], dtype=float)

    ep_l = sorted(epoch_lr)
    out["lr_epochs"] = np.array(ep_l, dtype=int)
    out["lr"]        = np.array([epoch_lr[e] for e in ep_l], dtype=float)

    val_raw: dict = {}
    for vk in all_val_keys:
        pairs = sorted([(ep, v) for ep in epochs for v in epoch_val[ep].get(vk, [])])
        if pairs:
            ri, rv = zip(*pairs)
            val_raw[vk] = (np.array(ri, dtype=int), np.array(rv, dtype=float))
    out["_val_raw"]    = val_raw

    return out


def parse_junyanz_log(path: str) -> dict:
    """Parse SR3 / Palette / junyanz-based training log."""
    iter_buf   = defaultdict(lambda: defaultdict(list))
    epoch_time: dict[int, int]   = {}
    epoch_lr:   dict[int, float] = {}
    last_ep = None

    with open(path, errors="replace") as fh:
        for line in fh:
            hdr = _JITER_RE.search(line)
            if hdr:
                epoch = int(hdr.group(1))
                for key, val in _JKV_RE.findall(line):
                    if key.lower() not in _JHEADER:
                        iter_buf[epoch][key].append(float(val))
                continue
            m = _JEPOCH_RE.search(line)
            if m:
                last_ep = int(m.group(1))
                epoch_time[last_ep] = int(m.group(2))
                continue
            m = _JLR_RE.search(line)
            if m and last_ep is not None and last_ep not in epoch_lr:
                epoch_lr[last_ep] = float(m.group(1))

    epochs = sorted(iter_buf.keys())
    if not epochs:
        raise ValueError(
            f"No iteration lines found in {path}. "
            "Check that the file uses junyanz or SinSR/ResShift format."
        )

    all_keys = sorted({k for ep in iter_buf.values() for k in ep})
    out: dict = {
        "epochs":     np.array(epochs, dtype=int),
        "x_label":    "Epoch",
        "_x_is_iter": False,
    }
    for key in all_keys:
        out[key] = np.array(
            [np.mean(iter_buf[ep][key]) if iter_buf[ep][key] else np.nan
             for ep in epochs],
            dtype=float,
        )

    ep_t = sorted(epoch_time)
    out["time_epochs"] = np.array(ep_t, dtype=int)
    out["time_sec"]    = np.array([epoch_time[e] for e in ep_t], dtype=float)

    ep_l = sorted(epoch_lr)
    out["lr_epochs"] = np.array(ep_l, dtype=int)
    out["lr"]        = np.array([epoch_lr[e] for e in ep_l], dtype=float)

    out["_val_raw"]    = {}

    return out


def parse_log(path: str, fmt: str | None = None) -> dict:
    """Parse log file. fmt is auto-detected if not provided."""
    if fmt is None:
        fmt = detect_format(path)
    if fmt == "sinsr":
        return parse_sinsr_log(path)
    if fmt == "resshift":
        return parse_resshift_log(path)
    if fmt == "palette":
        return parse_palette_log(path)
    return parse_junyanz_log(path)


# ---------------------------------------------------------------------------
# Key grouping — diffusion-centric
# ---------------------------------------------------------------------------

def _is_val_key(k: str) -> bool:
    return k.startswith("val_")


def group_loss_keys(all_keys_per_run: list[set]) -> list[tuple[str, list[str]]]:
    """
    Diffusion-centric grouping of training loss keys.

    Groups (in order):
        Diffusion loss       loss, l_pix, l_total, *diff*, *denoise*, *noise_pred*
        Distillation losses  mse, mse_gt, *distill*, *student*, *teacher*
        Other losses         everything else
    """
    union = sorted(
        k for s in all_keys_per_run for k in s
        if not _is_val_key(k) and k not in _META_KEYS
    )

    groups: dict[str, list[str]] = {
        "Diffusion loss":      [],
        "Distillation losses": [],
        "Other losses":        [],
    }

    for key in union:
        kl = key.lower()
        if (kl in ("loss", "l_pix", "l_total")
                or re.match(r"t\d+_loss$", kl)
                or any(x in kl for x in ("diff", "denoise", "noise_pred"))):
            groups["Diffusion loss"].append(key)
        elif (kl in ("mse", "mse_gt")
              or re.match(r"t\d+_mse$", kl)
              or any(x in kl for x in ("distill", "student", "teacher"))):
            groups["Distillation losses"].append(key)
        else:
            groups["Other losses"].append(key)

    return [(title, keys) for title, keys in groups.items() if keys]


def extract_val_groups(all_keys_per_run: list[set]) -> dict[str, list[str]]:
    """
    Separate validation keys (val_* prefix) into sub-groups for dedicated panels.

    Returns:
        lpips      -> val_* keys related to LPIPS
        psnr_ssim  -> val_* keys related to PSNR or SSIM
        other      -> remaining val_* keys
    """
    all_val = sorted(k for s in all_keys_per_run for k in s if _is_val_key(k))
    groups: dict[str, list[str]] = {"lpips": [], "psnr_ssim": [], "other": []}
    for k in all_val:
        kl = k.lower()
        if "lpips" in kl:
            groups["lpips"].append(k)
        elif "psnr" in kl or "ssim" in kl:
            groups["psnr_ssim"].append(k)
        else:
            groups["other"].append(k)
    return groups


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#9333ea",
    "#ea580c", "#0891b2", "#65a30d", "#db2777",
]
LINE_STYLES = ["-", "--", ":", "-."]


def smooth_curve(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Centred moving-average. Returns (index_array, smoothed_y)."""
    if window < 2 or len(y) < window:
        return np.arange(len(y)), y
    kernel = np.ones(window) / window
    y_sm   = np.convolve(y, kernel, mode="valid")
    offset = window // 2
    return np.arange(offset, offset + len(y_sm)), y_sm


def _style_ax(ax, title: str, x_label: str = "Epoch") -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    ax.grid(True, alpha=0.3, linewidth=0.5)


def _plot_train_lines(ax, datasets, labels, colors, keys, smooth) -> None:
    """Plot smoothed training loss curves for `keys` across all datasets."""
    for ki, key in enumerate(keys):
        ls = LINE_STYLES[ki % len(LINE_STYLES)]
        for run_data, label, color in zip(datasets, labels, colors):
            if key not in run_data:
                continue
            y_full = run_data[key]
            if np.all(np.isnan(y_full)):
                continue
            x_full = run_data["epochs"]
            ax.plot(x_full, y_full, color=color, alpha=0.12,
                    linewidth=0.7, linestyle=ls)
            x_idx, y_sm = smooth_curve(y_full, smooth)
            x_sm = x_full[x_idx[0]: x_idx[0] + len(y_sm)]
            ax.plot(x_sm, y_sm, color=color, linewidth=1.8,
                    linestyle=ls, label=f"{label}  {key}")


def _lower_is_better(key: str) -> bool:
    return "lpips" in key.lower()


def _plot_val_lines(ax, datasets, labels, colors, keys) -> None:
    """
    Plot validation metric curves with best-point star annotation.
    Val metrics are sparse (NaN where no validation ran), so raw points
    are shown with markers instead of smoothing.
    """
    for ki, key in enumerate(keys):
        ls = LINE_STYLES[ki % len(LINE_STYLES)]
        display_key = key.replace("val_", "")
        lib = _lower_is_better(display_key)

        for run_data, label, color in zip(datasets, labels, colors):
            if key not in run_data:
                continue
            y_full = run_data[key]
            x_full = run_data["epochs"]
            valid  = ~np.isnan(y_full)
            if not np.any(valid):
                continue
            ax.plot(x_full[valid], y_full[valid], color=color, linewidth=1.8,
                    linestyle=ls, marker="o", markersize=4,
                    label=f"{label}  {display_key}")

            # Best-point annotation using raw iteration/epoch positions
            val_raw    = run_data.get("_val_raw", {})
            x_is_iter  = run_data.get("_x_is_iter", False)

            if display_key in val_raw:
                raw_pos, raw_vals = val_raw[display_key]
            else:
                raw_pos  = x_full[valid]
                raw_vals = y_full[valid]

            if len(raw_vals) == 0:
                continue

            best_idx  = int(np.argmin(raw_vals) if lib else np.argmax(raw_vals))
            best_pos  = int(raw_pos[best_idx])
            best_val  = float(raw_vals[best_idx])
            best_x    = best_pos
            pos_label = f"iter {best_pos:,}" if x_is_iter else f"epoch {best_pos}"

            arrow = "↓" if lib else "↑"
            ax.plot(best_x, best_val, marker="*", markersize=13,
                    color=color, zorder=6,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.annotate(
                f"{arrow} {pos_label}\n{best_val:.4f}",
                xy=(best_x, best_val),
                xytext=(6, 8), textcoords="offset points",
                fontsize=7, color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          alpha=0.8, ec=color, lw=0.6),
            )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def make_figure(
    datasets:    list[dict],
    labels:      list[str],
    loss_groups: list[tuple[str, list[str]]],
    val_groups:  dict[str, list[str]],
    smooth:      int,
) -> plt.Figure:
    has_lr        = any(len(d.get("lr",       [])) > 0 for d in datasets)
    has_time      = any(len(d.get("time_sec", [])) > 0 for d in datasets)
    has_lpips     = bool(val_groups.get("lpips"))
    has_psnr_ssim = bool(val_groups.get("psnr_ssim"))
    has_val_other = bool(val_groups.get("other"))

    n_panels = (len(loss_groups)
                + int(has_lpips) + int(has_psnr_ssim) + int(has_val_other)
                + int(has_lr) + int(has_time))
    n_panels = max(n_panels, 1)

    fig, axes = plt.subplots(n_panels, 1, figsize=(11, max(3.0, 3.5 * n_panels)),
                             squeeze=False)
    axes  = [ax for row in axes for ax in row]
    colors = COLORS[: len(datasets)]
    ax_idx = 0

    # Use the first dataset's x_label (all should match if same format)
    x_label = datasets[0].get("x_label", "Epoch") if datasets else "Epoch"

    # -- Training loss group panels --
    for group_title, keys in loss_groups:
        ax = axes[ax_idx]; ax_idx += 1
        _style_ax(ax, group_title, x_label)
        ax.set_ylabel("Loss", fontsize=9)
        _plot_train_lines(ax, datasets, labels, colors, keys, smooth)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)

    # -- Val LPIPS panel --
    if has_lpips:
        ax = axes[ax_idx]; ax_idx += 1
        _style_ax(ax, "Validation LPIPS  (↓ better)", x_label)
        ax.set_ylabel("LPIPS", fontsize=9)
        _plot_val_lines(ax, datasets, labels, colors, val_groups["lpips"])
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)

    # -- Val PSNR / SSIM panel --
    if has_psnr_ssim:
        ax = axes[ax_idx]; ax_idx += 1
        _style_ax(ax, "Validation PSNR / SSIM  (↑ better)", x_label)
        ax.set_ylabel("dB / score", fontsize=9)
        _plot_val_lines(ax, datasets, labels, colors, val_groups["psnr_ssim"])
        ax.legend(fontsize=8, loc="lower right", framealpha=0.7)

    # -- Other val metrics --
    if has_val_other:
        ax = axes[ax_idx]; ax_idx += 1
        _style_ax(ax, "Other validation metrics", x_label)
        ax.set_ylabel("Score", fontsize=9)
        _plot_val_lines(ax, datasets, labels, colors, val_groups["other"])
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)

    # -- Learning rate panel --
    if has_lr:
        ax = axes[ax_idx]; ax_idx += 1
        _style_ax(ax, "Learning rate", x_label)
        ax.set_ylabel("LR", fontsize=9)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        for run_data, label, color in zip(datasets, labels, colors):
            if len(run_data.get("lr", [])) == 0:
                continue
            ax.plot(run_data["lr_epochs"], run_data["lr"],
                    color=color, linewidth=1.8,
                    marker=".", markersize=3, label=label)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)

    # -- Wall-clock time panel --
    if has_time:
        ax = axes[ax_idx]; ax_idx += 1
        _style_ax(ax, "Epoch wall-clock time", x_label)
        ax.set_ylabel("sec", fontsize=9)
        for run_data, label, color in zip(datasets, labels, colors):
            if len(run_data.get("time_sec", [])) == 0:
                continue
            ax.plot(run_data["time_epochs"], run_data["time_sec"],
                    color=color, linewidth=1.5,
                    marker=".", markersize=3, label=label)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)

    fig.suptitle(
        f"Training curves  |  {'  |  '.join(labels)}", fontsize=12, y=1.002
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def build_summary(
    datasets: list[dict],
    labels:   list[str],
    last_n:   int,
) -> list[dict]:
    all_metric_keys = sorted(
        k for d in datasets for k in d if k not in _META_KEYS
    )

    rows = []
    for run_data, label in zip(datasets, labels):
        epochs     = run_data["epochs"]
        x_is_iter  = run_data.get("_x_is_iter", False)
        total_sec  = float(np.sum(run_data.get("time_sec", [])))
        mean_sec   = (float(np.mean(run_data["time_sec"]))
                      if len(run_data.get("time_sec", [])) > 0 else float("nan"))

        step_unit = "iteration" if x_is_iter else "epoch"

        row: dict = {
            "label":               label,
            "steps_completed":     int(epochs[-1]) if len(epochs) > 0 else 0,
            "step_unit":           step_unit,
            "total_train_time_h":  round(total_sec / 3600, 3),
            "mean_time_per_step_sec": round(mean_sec, 1),
        }

        window = last_n if last_n > 0 else len(epochs)
        for key in all_metric_keys:
            if key not in run_data:
                row[f"final_{key}"] = ""
                continue
            vals  = run_data[key]
            tail  = vals[-window:] if window <= len(vals) else vals
            valid = tail[~np.isnan(tail)]
            row[f"final_{key}"] = (
                round(float(np.mean(valid)), 4) if len(valid) > 0 else ""
            )
        rows.append(row)
    return rows


def write_summary_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training curve plotter for diffusion virtual staining models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logs", nargs="+", required=True, metavar="FILE",
        help="SLURM .out or training log files, one per run.",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None, metavar="LABEL",
        help="Display labels matching --logs order. Defaults to file stems.",
    )
    parser.add_argument(
        "--name", default="training",
        help="Base name for output files (default: training).",
    )
    parser.add_argument(
        "--out-dir", default=".", metavar="DIR",
        help="Output directory. Created if absent.",
    )
    parser.add_argument(
        "--smooth", type=int, default=5,
        help="Moving-average window for training loss curves (default: 5).",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output image DPI (default: 150).",
    )
    parser.add_argument(
        "--last-n", type=int, default=10, metavar="N",
        help="Final N epochs for summary averages (default: 10). "
             "Use 0 for all epochs.",
    )
    args = parser.parse_args()

    labels = args.labels if args.labels else [Path(p).stem for p in args.logs]
    if len(labels) != len(args.logs):
        parser.error("--labels count must match --logs count.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets:     list[dict] = []
    all_key_sets: list[set]  = []

    for path, label in zip(args.logs, labels):
        fmt = detect_format(path)
        print(f"Parsing  {path}  ({label})  [format: {fmt}] ...",
              end=" ", flush=True)
        try:
            d = parse_log(path, fmt)
        except ValueError as exc:
            print(f"\nERROR: {exc}")
            raise SystemExit(1)
        x_unit   = "iterations" if d.get("_x_is_iter") else "epochs"
        n_steps  = int(d["epochs"][-1]) if len(d["epochs"]) > 0 else 0
        metric_keys = [k for k in d if k not in _META_KEYS]
        print(f"{n_steps} {x_unit},  keys: {', '.join(sorted(metric_keys))}")
        datasets.append(d)
        all_key_sets.append(set(metric_keys))

    loss_groups = group_loss_keys(all_key_sets)
    val_groups  = extract_val_groups(all_key_sets)

    if not loss_groups and not any(val_groups.values()):
        print("WARNING: No loss or validation keys detected. "
              "Check that the log files use a supported format.")

    fig = make_figure(datasets, labels, loss_groups, val_groups, args.smooth)

    plot_path = out_dir / f"{args.name}_training_curves.png"
    fig.savefig(plot_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved   : {plot_path}")

    summary_rows = build_summary(datasets, labels, args.last_n)
    csv_path = out_dir / f"{args.name}_training_summary.csv"
    write_summary_csv(summary_rows, csv_path)
    print(f"Summary CSV  : {csv_path}")

    print(f"\n{'Label':<24} {'Steps':>7} {'Unit':<12} {'Total (h)':>10} {'Sec/step':>10}")
    print("-" * 68)
    for row in summary_rows:
        print(f"{row['label']:<24} {row['steps_completed']:>7} "
              f"{row['step_unit']:<12} "
              f"{row['total_train_time_h']:>10.2f} "
              f"{row['mean_time_per_step_sec']:>10.1f}")

    # Best validation checkpoint summary
    print()
    for run_data, label in zip(datasets, labels):
        val_raw   = run_data.get("_val_raw", {})
        x_is_iter = run_data.get("_x_is_iter", False)
        if not val_raw:
            continue
        print(f"Best validation checkpoints  [{label}]:")
        for vk in sorted(val_raw):
            raw_pos, raw_vals = val_raw[vk]
            lib      = _lower_is_better(vk)
            best_idx = int(np.argmin(raw_vals) if lib else np.argmax(raw_vals))
            best_pos = int(raw_pos[best_idx])
            best_val = float(raw_vals[best_idx])
            arrow    = "↓" if lib else "↑"
            pos_str  = f"iter {best_pos:,}" if x_is_iter else f"epoch {best_pos}"
            print(f"  {vk:<12} {arrow}  {best_val:.4f}  at {pos_str}")


if __name__ == "__main__":
    main()