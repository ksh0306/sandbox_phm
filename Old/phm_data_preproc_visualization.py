"""phm_data_preproc.py 출력 h5를 효과적으로 시각화.

데이터가 매우 빽빽하므로 단순 line plot 대신:
  - 시계열: 픽셀 단위 min/max envelope (피크 보존)
  - 청크 FFT: (시간 × 주파수) 스펙트로그램 (log color scale)
"""
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def envelope_minmax(x, target_width):
    """1D 시계열을 target_width 픽셀로 (min, max) envelope downsampling.

    각 픽셀 기둥마다 해당 구간의 (min, max)를 반환하여 fill_between으로
    채우면 spike/peak이 사라지지 않는다.

    Returns: (sample_centers, mn, mx)  shape: (target_width,) each.
    """
    n = len(x)
    if n <= target_width * 2:
        idx = np.arange(n, dtype=np.float64)
        return idx, x.astype(np.float64), x.astype(np.float64)
    bin_size = n // target_width
    usable = bin_size * target_width
    reshaped = np.asarray(x[:usable]).reshape(target_width, bin_size)
    mn = reshaped.min(axis=1).astype(np.float64)
    mx = reshaped.max(axis=1).astype(np.float64)
    centers = np.arange(target_width, dtype=np.float64) * bin_size + bin_size / 2.0
    return centers, mn, mx


def _log_norm(spec, floor=1e-12):
    """LogNorm용 vmin/vmax. 0/음수는 floor로 클램프."""
    pos = spec[spec > 0]
    if pos.size == 0:
        return mcolors.LogNorm(vmin=floor, vmax=floor * 10)
    vmin = max(np.percentile(pos, 1), floor)
    vmax = pos.max()
    if vmax <= vmin:
        vmax = vmin * 10
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)


def plot_bundle(in_path, out_path, t_target_pixels=1200):
    with h5py.File(in_path, "r") as f:
        raw = f["raw"][:]                        # (6, N)
        scaled = f["scaled"][:]                  # (6, N)
        raw_fft = f["raw_fft"][:]                # (6, n_chunks, n_bins)
        scaled_fft = f["scaled_fft"][:]
        fft_freq = f["fft_freq"][:]
        fs_hz = float(f.attrs["fs_hz"])
        window_ms = float(f.attrs["window_ms"])
        channels = [
            c.decode() if isinstance(c, bytes) else c
            for c in f.attrs["channels"]
        ]

    n_chans, n_samples = raw.shape
    n_chunks = raw_fft.shape[1]
    duration_s = n_samples / fs_hz
    # 청크 시간축: 각 청크의 중심 [s]
    chunk_t = (np.arange(n_chunks) + 0.5) * (window_ms / 1000.0)

    fig, axes = plt.subplots(
        n_chans, 4,
        figsize=(22, 14),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.4, 1.0, 1.4, 1.0]},
    )

    for i in range(n_chans):
        kind = "Voltage" if i < 3 else "Current"
        ax_raw, ax_raw_spec, ax_log, ax_log_spec = axes[i]

        # --- col 1: raw min/max envelope ---
        idx, mn, mx = envelope_minmax(raw[i], t_target_pixels)
        t = idx / fs_hz
        ax_raw.fill_between(t, mn, mx, lw=0, alpha=0.85, color="tab:blue")
        ax_raw.set_ylabel(f"{channels[i]}\n({kind})")
        ax_raw.grid(True, alpha=0.3)
        ax_raw.set_xlim(0, duration_s)
        if i == 0:
            ax_raw.set_title("Raw — min/max envelope")

        # --- col 2: raw spectrogram (time × freq, log magnitude) ---
        spec = raw_fft[i].T  # (n_bins, n_chunks)
        ax_raw_spec.pcolormesh(
            chunk_t, fft_freq, spec,
            shading="auto",
            norm=_log_norm(spec),
            cmap="magma",
        )
        ax_raw_spec.set_xlim(0, duration_s)
        ax_raw_spec.set_ylim(fft_freq[0], fft_freq[-1])
        if i == 0:
            ax_raw_spec.set_title("Raw spectrogram (log mag)")

        # --- col 3: log16 envelope ---
        idx, mn, mx = envelope_minmax(scaled[i], t_target_pixels)
        t = idx / fs_hz
        ax_log.fill_between(t, mn, mx, lw=0, alpha=0.85, color="tab:orange")
        ax_log.grid(True, alpha=0.3)
        ax_log.set_xlim(0, duration_s)
        if i == 0:
            ax_log.set_title("log16 — min/max envelope")

        # --- col 4: log16 spectrogram ---
        spec = scaled_fft[i].T
        ax_log_spec.pcolormesh(
            chunk_t, fft_freq, spec,
            shading="auto",
            norm=_log_norm(spec),
            cmap="magma",
        )
        ax_log_spec.set_xlim(0, duration_s)
        ax_log_spec.set_ylim(fft_freq[0], fft_freq[-1])
        if i == 0:
            ax_log_spec.set_title("log16 spectrogram (log mag)")

    # 마지막 행에 x축 라벨
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    axes[-1, 2].set_xlabel("time [s]")
    axes[-1, 3].set_xlabel("time [s]")
    # 각 행의 spectrogram에 freq 축 라벨
    for r in range(n_chans):
        axes[r, 1].set_ylabel("freq [Hz]")
        axes[r, 3].set_ylabel("freq [Hz]")

    fig.suptitle(
        f"{Path(in_path).stem}  "
        f"(fs={fs_hz:.0f} Hz, N={n_samples} samples, duration={duration_s:.1f}s, "
        f"window={window_ms:.0f}ms, chunks={n_chunks})",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_xcorr(in_path, out_path):
    """상별 (v_k ★ i_k) 청크 교차상관과 그 FFT를 heatmap으로 시각화.

    - 좌: lag-time heatmap (signed value, diverging cmap, ±99pct 클립)
    - 우: xcorr_fft 스펙트로그램 (log magnitude)
    """
    with h5py.File(in_path, "r") as f:
        xcorr = f["xcorr"][:]                   # (3, n_chunks, 2W-1)
        xcorr_fft = f["xcorr_fft"][:]           # (3, n_chunks, W)
        xcorr_lag_s = f["xcorr_lag_s"][:]       # (2W-1,)
        xcorr_fft_freq = f["xcorr_fft_freq"][:] # (W,)
        fs_hz = float(f.attrs["fs_hz"])
        window_ms = float(f.attrs["window_ms"])
        pair_names = [
            c.decode() if isinstance(c, bytes) else c
            for c in f.attrs["xcorr_pairs"]
        ]

    n_pairs, n_chunks, _ = xcorr.shape
    duration_s = n_chunks * (window_ms / 1000.0)
    chunk_t = (np.arange(n_chunks) + 0.5) * (window_ms / 1000.0)
    lag_ms = xcorr_lag_s * 1000.0

    fig, axes = plt.subplots(
        n_pairs, 2,
        figsize=(18, 9),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    if n_pairs == 1:
        axes = axes.reshape(1, 2)

    for k in range(n_pairs):
        ax_xc, ax_xf = axes[k]

        # --- left: xcorr lag-time heatmap (signed) ---
        spec = xcorr[k].T  # (n_lag, n_chunks)
        vmax = float(np.percentile(np.abs(spec), 99)) or 1.0
        im_xc = ax_xc.pcolormesh(
            chunk_t, lag_ms, spec,
            shading="auto",
            vmin=-vmax, vmax=vmax,
            cmap="RdBu_r",
        )
        ax_xc.set_ylabel(f"{pair_names[k]}\nlag [ms]")
        ax_xc.set_xlim(0, duration_s)
        ax_xc.axhline(0, color="k", lw=0.4, alpha=0.5)
        fig.colorbar(im_xc, ax=ax_xc, fraction=0.025, pad=0.01)
        if k == 0:
            ax_xc.set_title("xcorr lag-time  (signed, ±99pct clip)")

        # --- right: xcorr_fft spectrogram (log magnitude) ---
        spec_f = xcorr_fft[k].T  # (W, n_chunks)
        im_xf = ax_xf.pcolormesh(
            chunk_t, xcorr_fft_freq, spec_f,
            shading="auto",
            norm=_log_norm(spec_f),
            cmap="magma",
        )
        ax_xf.set_xlim(0, duration_s)
        ax_xf.set_ylim(xcorr_fft_freq[0], xcorr_fft_freq[-1])
        ax_xf.set_ylabel("freq [Hz]")
        fig.colorbar(im_xf, ax=ax_xf, fraction=0.025, pad=0.01)
        if k == 0:
            ax_xf.set_title("xcorr FFT spectrogram  (log mag)")

    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")

    fig.suptitle(
        f"{Path(in_path).stem} — xcorr  "
        f"(fs={fs_hz:.0f}Hz, window={window_ms:.0f}ms, "
        f"lag range=±{lag_ms.max():.1f}ms, chunks={n_chunks})",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_xcorr_lines(in_path, out_path):
    """xcorr / xcorr_fft 합성 데이터를 라인 그래프로 요약 (heatmap 보완용).

    상별 3행 × 3열:
      - col 1: 시간평균 xcorr 프로파일 (lag vs value)
      - col 2: 청크별 peak lag (ms) + peak |xcorr| (log) 시간 변화
      - col 3: 시간평균 xcorr FFT 스펙트럼 (freq vs |X|)
    """
    with h5py.File(in_path, "r") as f:
        xcorr = f["xcorr"][:]                   # (3, n_chunks, 2W-1)
        xcorr_fft = f["xcorr_fft"][:]           # (3, n_chunks, W)
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
    chunk_t = (np.arange(n_chunks) + 0.5) * (window_ms / 1000.0)
    lag_ms = xcorr_lag_s * 1000.0

    # 시간평균 통계 (전체 청크에 대한 평균)
    xcorr_mean_profile = xcorr.mean(axis=1)         # (3, 2W-1)
    xcorr_fft_mean = xcorr_fft.mean(axis=1)         # (3, W)

    # 청크별 통계: peak |xcorr|와 그 lag
    abs_xcorr = np.abs(xcorr)
    peak_idx = np.argmax(abs_xcorr, axis=2)         # (3, n_chunks)
    peak_lag_ms = lag_ms[peak_idx]                  # (3, n_chunks)
    peak_mag = np.take_along_axis(
        abs_xcorr, peak_idx[:, :, None], axis=2
    ).squeeze(2)                                    # (3, n_chunks)

    fig, axes = plt.subplots(
        n_pairs, 3,
        figsize=(20, 9),
        constrained_layout=True,
    )
    if n_pairs == 1:
        axes = axes.reshape(1, 3)

    for k in range(n_pairs):
        ax_p, ax_pk, ax_s = axes[k]

        # --- col 1: 시간평균 xcorr 프로파일 ---
        ax_p.plot(lag_ms, xcorr_mean_profile[k], lw=0.7, color="tab:blue")
        ax_p.axvline(0, color="k", lw=0.4, alpha=0.5)
        ax_p.grid(True, alpha=0.3)
        ax_p.set_ylabel(f"{pair_names[k]}\nxcorr (mean)")
        if k == 0:
            ax_p.set_title("time-averaged xcorr profile")
        if k == n_pairs - 1:
            ax_p.set_xlabel("lag [ms]")

        # --- col 2: peak lag + peak magnitude 시간 변화 ---
        ax_pk.plot(chunk_t, peak_lag_ms[k], lw=0.6, color="tab:red")
        ax_pk.set_ylabel("peak lag [ms]", color="tab:red")
        ax_pk.tick_params(axis="y", labelcolor="tab:red")
        ax_pk.grid(True, alpha=0.3)
        ax_pk.set_xlim(0, duration_s)
        ax_pk.axhline(0, color="k", lw=0.4, alpha=0.3)
        ax_pk2 = ax_pk.twinx()
        ax_pk2.plot(chunk_t, peak_mag[k], lw=0.6, color="tab:gray", alpha=0.7)
        ax_pk2.set_ylabel("peak |xcorr|", color="tab:gray")
        ax_pk2.set_yscale("log")
        if k == 0:
            ax_pk.set_title("peak lag (red) & peak |xcorr| (gray, log)")
        if k == n_pairs - 1:
            ax_pk.set_xlabel("time [s]")

        # --- col 3: 시간평균 xcorr FFT 스펙트럼 ---
        ax_s.plot(xcorr_fft_freq, xcorr_fft_mean[k], lw=0.7, color="tab:purple")
        ax_s.set_yscale("log")
        ax_s.grid(True, which="both", alpha=0.3)
        ax_s.set_ylabel("xcorr_fft (mean)")
        if k == 0:
            ax_s.set_title("time-averaged xcorr FFT spectrum (log)")
        if k == n_pairs - 1:
            ax_s.set_xlabel("freq [Hz]")

    fig.suptitle(
        f"{Path(in_path).stem} — xcorr summary lines  "
        f"(fs={fs_hz:.0f}Hz, window={window_ms:.0f}ms, "
        f"chunks={n_chunks}, lag range=±{lag_ms.max():.1f}ms)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved: {out_path}")


# 입력 후보: 4토큰(motor_id_date_seq.h5)이 아니라 3토큰(motor_id_date.h5) 형태.
_INPUT_PATTERN = "motor_*.h5"


def _is_preproc_output(path):
    parts = path.stem.split("_")
    return len(parts) == 3  # motor / id / date


def main():
    if len(sys.argv) >= 2:
        in_paths = [Path(p) for p in sys.argv[1:]]
    else:
        in_paths = sorted(
            fp for fp in Path(".").glob(_INPUT_PATTERN) if _is_preproc_output(fp)
        )
    if not in_paths:
        print("no preprocessed h5 files found (expected name: motor_<id>_<date>.h5)")
        return
    for fp in in_paths:
        plot_bundle(fp, fp.with_name(f"{fp.stem}_viz.png"))
        plot_xcorr(fp, fp.with_name(f"{fp.stem}_xcorr_viz.png"))
        plot_xcorr_lines(fp, fp.with_name(f"{fp.stem}_xcorr_lines.png"))


if __name__ == "__main__":
    main()
