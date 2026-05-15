"""phm_data_preproc.py 출력 h5의 fft_xcorr / scaled_xcorr 시각화.

데이터 의미 요약:

  scaled_xcorr (6, n_chunks, 2W-1)  —  시간 도메인 청크 cross-correlation
    입력: scaled = log16p(raw) (0~1 정규화) 의 W=1000 샘플(50ms) 청크.
    6 페어 (v1-v2, v2-v3, v3-v1, i1-i2, i2-i3, i3-i1)에 대해
      r[m] = Σ_n a[n] · b[n+m],  m = -(W-1) .. +(W-1)
    lag 축은 시간(초). 3상에서 위상이 120° 차이 나면 60Hz 기준 ≈5.56ms,
    50Hz 기준 ≈6.67ms 위치에 사인성 피크가 보임. 입력이 non-negative이므로
    중심에 큰 삼각형 envelope이 깔리고 그 위에 위상-사인 패턴이 얹힘.
    청크 축을 보면 시간에 따른 위상 관계 변화(부하 변동, 위상 손실 등)를 추적.

  fft_xcorr (6, 2(N//2+1)-1)  —  주파수 도메인 cross-correlation
    입력: 전체 신호의 FFT magnitude (scaled_fft, shape (6, N//2+1)).
    같은 6 페어에 대해 magnitude 스펙트럼을 1차원 시퀀스로 보고 xcorr.
    lag 축은 주파수 시프트 [Hz], 간격 = FS_HZ/N (전체 길이 기반).
      - lag=0 값 = 두 스펙트럼의 inner product → 스펙트럼 모양 유사도
      - 비-0 lag 피크 = 한 채널 스펙트럼이 다른 채널 대비 그만큼 시프트된 성분
      - 피크가 분산/평탄 = 스펙트럼 모양이 서로 다름 (고조파 비대칭/결함)

레이아웃: 6행(페어) × 3열
  col 1: scaled_xcorr 히트맵 (chunk-time × lag-time, log color)
  col 2: scaled_xcorr 시간평균 프로파일 (lag-time vs <r>_t)
  col 3: fft_xcorr 라인 (freq lag vs r, log y)
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
    """LogNorm vmin/vmax. 0/음수는 floor, 하한은 양수값의 percentile_lo퍼센타일."""
    pos = spec[spec > 0]
    if pos.size == 0:
        return mcolors.LogNorm(vmin=floor, vmax=floor * 10)
    vmin = max(np.percentile(pos, percentile_lo), floor)
    vmax = float(pos.max())
    if vmax <= vmin:
        vmax = vmin * 10
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)


def plot_corr(
    in_path,
    out_path,
    *,
    lag_zoom_ms=None,
    freq_lag_zoom_hz=None,
):
    """fft_xcorr와 scaled_xcorr를 페어별 (6행 × 3열) 로 시각화.

    파라미터:
      lag_zoom_ms      : scaled_xcorr y축(시간 lag) 줌 범위 [ms].
                          예: 20 → ±20ms. None이면 전체(±(W-1)/Fs ms).
      freq_lag_zoom_hz : fft_xcorr x축(주파수 lag) 줌 범위 [Hz].
                          예: 1000 → ±1000Hz. None이면 전체.
    """
    with h5py.File(in_path, "r") as f:
        fft_xcorr = f["fft_xcorr"][:]                  # (6, 2(N//2+1)-1)
        scaled_xcorr = f["scaled_xcorr"][:]            # (6, n_chunks, 2W-1)
        fft_xcorr_lag_hz = f["fft_xcorr_lag_hz"][:]
        scaled_xcorr_lag_s = f["scaled_xcorr_lag_s"][:]
        window_ms = float(f.attrs["window_ms"])
        W = int(f.attrs["samples_per_window"])
        pair_names = [
            c.decode() if isinstance(c, bytes) else c
            for c in f.attrs["xcorr_pairs"]
        ]

    n_pairs, n_chunks, _ = scaled_xcorr.shape
    duration_s = n_chunks * (window_ms / 1000.0)
    lag_ms = scaled_xcorr_lag_s * 1000.0

    # 시간 평균 프로파일 (페어별 1D)
    scaled_xcorr_mean = scaled_xcorr.mean(axis=1)      # (6, 2W-1)

    fig, axes = plt.subplots(
        n_pairs, 3,
        figsize=(20, 14),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.4, 1.0, 1.4]},
    )
    if n_pairs == 1:
        axes = axes.reshape(1, 3)

    for k in range(n_pairs):
        ax_hm, ax_prof, ax_fft = axes[k]

        # col 1: scaled_xcorr heatmap (chunk_time × lag_ms, log color)
        spec = scaled_xcorr[k].T                       # (n_lag, n_chunks)
        im_hm = ax_hm.imshow(
            spec,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[0.0, duration_s, float(lag_ms[0]), float(lag_ms[-1])],
            norm=_log_norm(spec),
            cmap="magma",
        )
        ax_hm.set_ylabel(f"{pair_names[k]}\nlag [ms]")
        ax_hm.axhline(0, color="cyan", lw=0.4, alpha=0.5)
        if lag_zoom_ms is not None:
            ax_hm.set_ylim(-lag_zoom_ms, lag_zoom_ms)
        fig.colorbar(im_hm, ax=ax_hm, fraction=0.025, pad=0.01)
        if k == 0:
            ax_hm.set_title("scaled_xcorr heatmap  (log color, chunk × lag-time)")

        # col 2: scaled_xcorr 시간평균 프로파일
        ax_prof.plot(lag_ms, scaled_xcorr_mean[k], lw=0.7, color="tab:blue")
        ax_prof.axvline(0, color="k", lw=0.4, alpha=0.5)
        ax_prof.grid(True, alpha=0.3)
        if lag_zoom_ms is not None:
            ax_prof.set_xlim(-lag_zoom_ms, lag_zoom_ms)
        else:
            ax_prof.set_xlim(float(lag_ms[0]), float(lag_ms[-1]))
        if k == 0:
            ax_prof.set_title("scaled_xcorr  time-averaged profile")

        # col 3: fft_xcorr 라인 (freq lag vs r, log y)
        ax_fft.plot(fft_xcorr_lag_hz, fft_xcorr[k], lw=0.5, color="tab:purple")
        ax_fft.axvline(0, color="k", lw=0.4, alpha=0.5)
        ax_fft.set_yscale("log")
        ax_fft.grid(True, which="both", alpha=0.3)
        if freq_lag_zoom_hz is not None:
            ax_fft.set_xlim(-freq_lag_zoom_hz, freq_lag_zoom_hz)
        else:
            ax_fft.set_xlim(
                float(fft_xcorr_lag_hz[0]), float(fft_xcorr_lag_hz[-1])
            )
        if k == 0:
            ax_fft.set_title("fft_xcorr  (whole-signal magnitude xcorr, log y)")

    axes[-1, 0].set_xlabel("chunk time [s]")
    axes[-1, 1].set_xlabel("lag [ms]")
    axes[-1, 2].set_xlabel("freq lag [Hz]")

    freq_bin_hz = float(fft_xcorr_lag_hz[1] - fft_xcorr_lag_hz[0])
    fig.suptitle(
        f"{Path(in_path).stem}  —  scaled_xcorr (time-domain) + fft_xcorr (freq-domain)  "
        f"(W={W}, n_chunks={n_chunks}, duration={duration_s:.2f}s, "
        f"freq lag bin = {freq_bin_hz:.3f} Hz)",
        fontsize=11,
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
        # lag_zoom_ms, freq_lag_zoom_hz를 조절해 0 근방 확대 가능.
        # 예) lag_zoom_ms=20 → ±20ms (60Hz 위상차 영역),
        #     freq_lag_zoom_hz=500 → ±500Hz (전력 기본주파수~수배음)
        plot_corr(
            fp,
            fp.with_name(f"{fp.stem}_corr_view.png"),
            lag_zoom_ms=None,
            freq_lag_zoom_hz=None,
        )


if __name__ == "__main__":
    main()
