"""phm_data_preproc.py 출력 h5의 시각화.

시간 도메인(raw, scaled)은 '값이 급변하는 구간' 주변 n_windows × W 샘플만
잘라서 표시한다. FFT(raw_fft, scaled_fft)는 전처리에서 전체 신호 1회로
계산되어 있으므로 burst 영역과 무관하게 채널별 전체 스펙트럼을 그린다
(저주파까지 해상도 = FS_HZ/N).

함수 파라미터 (plot_burst_view):
  - n_windows       : 시간 도메인 표시 폭 = n_windows × W (소수 허용, default 4)
  - center_sample   : 표시 중심 (절대 샘플 인덱스). None이면 자동 검출.
  - detect_channel  : 자동 검출 기준 채널 인덱스(0..5). None이면 전 채널 합산.
"""
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_burst_center(x, win_size, channel=None):
    """(C, N) 시계열에서 값이 급변하는 구간의 중심 샘플 인덱스를 찾는다.

    |diff(x)|를 길이 win_size 박스카로 컨볼브한 뒤 최댓값 위치를 반환.
    channel=None 이면 전 채널 |diff| 를 채널 축으로 합산해서 한 신호로 만든다.
    """
    n_chans, n = x.shape
    if n < 2:
        return 0
    diff = np.abs(np.diff(x.astype(np.float64), axis=1))
    agg = diff[channel] if channel is not None else diff.sum(axis=0)
    if len(agg) < win_size:
        return int(np.argmax(agg))
    kernel = np.ones(win_size, dtype=np.float64) / win_size
    smoothed = np.convolve(agg, kernel, mode="valid")
    # smoothed[k] 는 diff[k:k+win_size] 평균 → 원 시계열의 [k+1, k+win_size] 구간.
    # 중심은 k + win_size//2 로 환산.
    return int(np.argmax(smoothed)) + win_size // 2


def plot_burst_view(
    in_path,
    out_path,
    *,
    n_windows=4,
    center_sample=None,
    detect_channel=None,
):
    """raw/scaled 시간 영역은 '급변 구간' n_windows×W 만, raw_fft/scaled_fft는
    전체 신호 스펙트럼을 그린다.

    레이아웃: 6행(채널) × 4열 (raw 시간 / raw_fft 전체 / scaled 시간 / scaled_fft 전체).
    """
    with h5py.File(in_path, "r") as f:
        raw = f["raw"][:]
        scaled = f["scaled"][:]
        raw_fft = f["raw_fft"][:]                # (6, N//2+1)
        scaled_fft = f["scaled_fft"][:]          # (6, N//2+1)
        fft_freq = f["fft_freq"][:]              # (N//2+1,)
        fs_hz = float(f.attrs["fs_hz"])
        W = int(f.attrs["samples_per_window"])
        channels = [
            c.decode() if isinstance(c, bytes) else c
            for c in f.attrs["channels"]
        ]

    n_chans, n_samples = raw.shape

    # 시간 도메인 표시 폭(샘플): n_windows × W (소수 허용)
    span = max(1, int(round(n_windows * W)))
    span = min(span, n_samples)

    # 표시 중심 결정 (수동 우선, 미지정 시 자동 검출)
    if center_sample is None:
        center_sample = find_burst_center(raw, win_size=W, channel=detect_channel)

    # 가용 범위로 클램프 (청크 스냅 없음 — FFT는 전체 신호라 청크 정렬 불필요)
    half = span // 2
    start_raw = int(np.clip(center_sample - half, 0, max(0, n_samples - span)))
    end_raw = start_raw + span

    t_ms = np.arange(start_raw, end_raw) / fs_hz * 250.0

    # 표시 구간 내에 걸리는 W 경계 (scaled_xcorr 청크 경계, 시각적 참조용)
    def _draw_window_lines(ax):
        first_k = (start_raw + W - 1) // W
        last_k = end_raw // W
        for k in range(first_k, last_k + 1):
            ax.axvline(k * W / fs_hz * 250.0, color="k", lw=0.3, alpha=0.3)

    # FFT x축: 0 Hz는 log 스케일에서 제외, 첫 양수 빈부터 표시
    pos = fft_freq > 0
    fft_freq_pos = fft_freq[pos]

    fig, axes = plt.subplots(
        n_chans, 4,
        figsize=(22, 14),
        constrained_layout=True,
    )

    for i in range(n_chans):
        kind = "Voltage" if i < 3 else "Current"
        ax_raw, ax_rfft, ax_scaled, ax_sfft = axes[i]

        # col 1: raw time series (burst 영역)
        ax_raw.plot(t_ms, raw[i, start_raw:end_raw], lw=0.6, color="tab:blue")
        ax_raw.set_ylabel(f"{channels[i]}\n({kind})")
        ax_raw.grid(True, alpha=0.3)
        ax_raw.set_xlim(t_ms[0], t_ms[-1])
        _draw_window_lines(ax_raw)
        if i == 0:
            ax_raw.set_title(f"raw (burst region, span={span} samples)")

        # col 2: raw_fft — 전체 신호 스펙트럼 (log-log)
        ax_rfft.plot(fft_freq_pos, raw_fft[i, pos], lw=0.5, color="tab:blue")
        ax_rfft.set_xscale("log")
        ax_rfft.set_yscale("log")
        ax_rfft.grid(True, which="both", alpha=0.3)
        ax_rfft.set_xlim(fft_freq_pos[0], fft_freq_pos[-1])
        if i == 0:
            ax_rfft.set_title("raw_fft (whole signal, log-log)")

        # col 3: scaled time series (burst 영역)
        ax_scaled.plot(t_ms, scaled[i, start_raw:end_raw], lw=0.6, color="tab:orange")
        ax_scaled.grid(True, alpha=0.3)
        ax_scaled.set_xlim(t_ms[0], t_ms[-1])
        _draw_window_lines(ax_scaled)
        if i == 0:
            ax_scaled.set_title(f"scaled (burst region, span={span} samples)")

        # col 4: scaled_fft — 전체 신호 스펙트럼 (log-log)
        ax_sfft.plot(fft_freq_pos, scaled_fft[i, pos], lw=0.5, color="tab:orange")
        ax_sfft.set_xscale("log")
        ax_sfft.set_yscale("log")
        ax_sfft.grid(True, which="both", alpha=0.3)
        ax_sfft.set_xlim(fft_freq_pos[0], fft_freq_pos[-1])
        if i == 0:
            ax_sfft.set_title("scaled_fft (whole signal, log-log)")

    axes[-1, 0].set_xlabel("time [ms]")
    axes[-1, 1].set_xlabel("freq [Hz]")
    axes[-1, 2].set_xlabel("time [ms]")
    axes[-1, 3].set_xlabel("freq [Hz]")

    t0_ms = start_raw / fs_hz * 1000.0
    t1_ms = end_raw / fs_hz * 1000.0
    duration_s = n_samples / fs_hz
    fig.suptitle(
        f"{Path(in_path).stem}  —  burst-region (time) + whole-signal FFT  "
        f"(center≈sample {center_sample}, range=[{start_raw}, {end_raw})  "
        f"= [{t0_ms:.1f}, {t1_ms:.1f}] ms, "
        f"N={n_samples} ({duration_s:.2f}s), W={W}, n_windows={n_windows})",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(
        f"saved: {out_path}  "
        f"(center_sample={center_sample}, samples=[{start_raw},{end_raw}), "
        f"span={span})"
    )


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
        print("no preproces고sed h5 files found (expected name: motor_<id>_<date>.h5)")
        return
    for fp in in_paths:
        # 여기서 n_windows / center_sample / detect_channel 값을 바꿔가며 확인.
        plot_burst_view(
            fp,
            fp.with_name(f"{fp.stem}_burst_view.png"),
            n_windows=0.125,
            center_sample=None,
            detect_channel=None,
        )


if __name__ == "__main__":
    main()
