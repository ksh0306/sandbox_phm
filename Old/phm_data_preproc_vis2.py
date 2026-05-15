"""phm_data_preproc.py 출력 h5의 FFT 데이터를 비트맵(imshow) 형식으로 시각화.

raw_fft, scaled_fft (6, n_chunks, n_bins)을 채널별로 (시간 × 주파수) 2D 비트맵
으로 표현. interpolation='nearest'로 셀 = 픽셀 1:1 대응, 보간 없이 원본 충실도 유지.
"""
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def _log_norm(spec, floor=1e-12, percentile_lo=1.0):
    """LogNorm용 vmin/vmax. 0/음수는 floor, 하한은 양수값의 1퍼센타일."""
    pos = spec[spec > 0]
    if pos.size == 0:
        return mcolors.LogNorm(vmin=floor, vmax=floor * 10)
    vmin = max(np.percentile(pos, percentile_lo), floor)
    vmax = float(pos.max())
    if vmax <= vmin:
        vmax = vmin * 10
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)


def plot_fft_bitmap(in_path, out_path):
    with h5py.File(in_path, "r") as f:
        raw_fft = f["raw_fft"][:]                # (6, n_chunks, n_bins)
        scaled_fft = f["scaled_fft"][:]
        fft_freq = f["fft_freq"][:]
        fs_hz = float(f.attrs["fs_hz"])
        window_ms = float(f.attrs["window_ms"])
        channels = [
            c.decode() if isinstance(c, bytes) else c
            for c in f.attrs["channels"]
        ]

    n_chans, n_chunks, n_bins = raw_fft.shape
    duration_s = n_chunks * (window_ms / 1000.0)
    f_max = float(fft_freq[-1])

    fig, axes = plt.subplots(
        n_chans, 2,
        figsize=(16, 14),
        constrained_layout=True,
    )

    for i in range(n_chans):
        kind = "Voltage" if i < 3 else "Current"
        ax_raw, ax_log = axes[i]

        # raw_fft[i]: (n_chunks, n_bins) → 표시 시 freq를 y축, time을 x축
        # transpose해서 (n_bins, n_chunks) 후 origin='lower'로 freq 증가 방향 정상화.
        spec_raw = raw_fft[i].T
        im_raw = ax_raw.imshow(
            spec_raw,
            origin="lower",
            aspect="auto",
            interpolation="nearest",   # 픽셀 1:1 (보간 없음, 비트맵 충실)
            extent=[0.0, duration_s, 0.0, f_max],
            norm=_log_norm(spec_raw),
            cmap="magma",
        )
        ax_raw.set_ylabel(f"{channels[i]}\n({kind})\nfreq [Hz]")
        fig.colorbar(im_raw, ax=ax_raw, fraction=0.025, pad=0.01)
        if i == 0:
            ax_raw.set_title("Raw FFT bitmap  (log mag, nearest)")

        spec_log = scaled_fft[i].T
        im_log = ax_log.imshow(
            spec_log,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[0.0, duration_s, 0.0, f_max],
            norm=_log_norm(spec_log),
            cmap="magma",
        )
        fig.colorbar(im_log, ax=ax_log, fraction=0.025, pad=0.01)
        if i == 0:
            ax_log.set_title("log16 FFT bitmap  (log mag, nearest)")

    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")

    fig.suptitle(
        f"{Path(in_path).stem}  —  FFT bitmap "
        f"({n_chunks} chunks × {n_bins} bins, fs={fs_hz:.0f}Hz, "
        f"window={window_ms:.0f}ms, duration={duration_s:.1f}s)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_xcorr_bitmap(in_path, out_path):
    """xcorr / xcorr_fft를 imshow(nearest) 비트맵으로 시각화 (3 pairs × 2 cols)."""
    with h5py.File(in_path, "r") as f:
        xcorr = f["xcorr"][:]
        xcorr_fft = f["xcorr_fft"][:]
        xcorr_lag_s = f["xcorr_lag_s"][:]
        xcorr_fft_freq = f["xcorr_fft_freq"][:]
        fs_hz = float(f.attrs["fs_hz"])
        window_ms = float(f.attrs["window_ms"])
        pair_names = [
            c.decode() if isinstance(c, bytes) else c
            for c in f.attrs["xcorr_pairs"]
        ]

    n_pairs, n_chunks, _ = xcorr.shape
    duration_s = n_chunks * (window_ms / 1000.0)
    lag_ms_min = float(xcorr_lag_s[0]) * 1000.0
    lag_ms_max = float(xcorr_lag_s[-1]) * 1000.0
    f_max = float(xcorr_fft_freq[-1])

    fig, axes = plt.subplots(
        n_pairs, 2,
        figsize=(16, 9),
        constrained_layout=True,
    )
    if n_pairs == 1:
        axes = axes.reshape(1, 2)

    for k in range(n_pairs):
        ax_xc, ax_xf = axes[k]

        # xcorr bitmap (signed, diverging cmap)
        spec = xcorr[k].T  # (n_lag, n_chunks)
        vmax = float(np.percentile(np.abs(spec), 99)) or 1.0
        im_xc = ax_xc.imshow(
            spec,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[0.0, duration_s, lag_ms_min, lag_ms_max],
            vmin=-vmax, vmax=vmax,
            cmap="RdBu_r",
        )
        ax_xc.set_ylabel(f"{pair_names[k]}\nlag [ms]")
        ax_xc.axhline(0, color="k", lw=0.4, alpha=0.5)
        fig.colorbar(im_xc, ax=ax_xc, fraction=0.025, pad=0.01)
        if k == 0:
            ax_xc.set_title("xcorr bitmap  (signed, nearest, ±99pct)")

        # xcorr_fft bitmap (log magnitude)
        spec_f = xcorr_fft[k].T  # (W, n_chunks)
        im_xf = ax_xf.imshow(
            spec_f,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[0.0, duration_s, 0.0, f_max],
            norm=_log_norm(spec_f),
            cmap="magma",
        )
        ax_xf.set_ylabel("freq [Hz]")
        fig.colorbar(im_xf, ax=ax_xf, fraction=0.025, pad=0.01)
        if k == 0:
            ax_xf.set_title("xcorr FFT bitmap  (log mag, nearest)")

    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")

    fig.suptitle(
        f"{Path(in_path).stem} — xcorr bitmap  "
        f"(fs={fs_hz:.0f}Hz, window={window_ms:.0f}ms, "
        f"lag={lag_ms_min:.1f}~{lag_ms_max:.1f}ms, chunks={n_chunks})",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved: {out_path}")


def _is_preproc_output(path):
    return len(path.stem.split("_")) == 3


def main():
    if len(sys.argv) >= 2:
        in_paths = [Path(p) for p in sys.argv[1:]]
    else:
        in_paths = sorted(
            fp for fp in Path(".").glob("motor_*.h5") if _is_preproc_output(fp)
        )
    if not in_paths:
        print("no preprocessed h5 files found (expected name: motor_<id>_<date>.h5)")
        return
    for fp in in_paths:
        plot_fft_bitmap(fp, fp.with_name(f"{fp.stem}_fft_bitmap.png"))
        plot_xcorr_bitmap(fp, fp.with_name(f"{fp.stem}_xcorr_bitmap.png"))


if __name__ == "__main__":
    main()
