"""motorsig 시각화 모듈 — 단계 클래스 인스턴스를 그림으로 표현한다.

Old(v1)의 시각화 코드(`phm_data_vis_time.py` / `phm_data_vis_fft.py`)의
표현 방식을 v2 클래스 계층에 맞춰 옮겨 왔다:

* `plot_waveforms`   시간영역 6채널 파형 + cubic spline 보간 (Old plot_signals)
* `plot_fft_bars`    채널별 진폭 스펙트럼 막대그래프 (Old plot_fft_bars)
* `plot_xcorr_heatmaps`  채널쌍 상관 (행 × lag) 히트맵 그리드 (Old)
* `plot_xcorr_profiles`  채널쌍 상관의 행-평균 프로파일 그리드 (Old)

각 함수는 `matplotlib.figure.Figure`를 반환한다. PNG 저장은 `save_figure`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경에서 PNG 저장 (display 불필요)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

from .fastadc import FastAdcData  # noqa: E402
from .xcorr import CrossCorrLog  # noqa: E402

# cubic spline 보간 시 표본 사이를 몇 배로 촘촘히 그릴지 (Old: SPLINE_OVERSAMPLE).
SPLINE_OVERSAMPLE = 8


def _flatten(signal: FastAdcData) -> np.ndarray:
    """(패킷, 채널, 샘플) → (채널, 패킷*샘플) 연속 신호로 평탄화."""
    d = signal.data
    return d.transpose(1, 0, 2).reshape(d.shape[1], -1)


def _draw_channel(ax, t: np.ndarray, y: np.ndarray, spline: bool) -> None:
    """한 채널 시계열을 표본점 + cubic spline 보간 곡선으로 그린다."""
    if spline and len(t) >= 4:
        ax.plot(t, y, "o", markersize=2.0, color="C0", label="sample")
        t_dense = np.linspace(t[0], t[-1], (len(t) - 1) * SPLINE_OVERSAMPLE + 1)
        ax.plot(
            t_dense, CubicSpline(t, y)(t_dense),
            "-", linewidth=0.8, color="C1", label="spline",
        )
    else:
        ax.plot(t, y, "-", linewidth=0.8, color="C0")
    ax.grid(True, alpha=0.3)


def plot_waveforms(
    signals: FastAdcData | list[FastAdcData],
    *,
    labels: list[str] | None = None,
    start: int = 0,
    length: int = 500,
    channels: tuple[str, ...] | None = None,
    spline: bool = True,
    title: str | None = None,
):
    """FastAdcData / LogNormalized 시간영역 파형 (6채널 × 신호 수).

    각 신호를 한 열로 나란히 그려 비교한다(예: raw vs log 정규화). 채널마다
    표본점과 cubic spline 보간 곡선을 함께 표시한다(Old의 `plot_signals`).
    """
    items = signals if isinstance(signals, list) else [signals]
    if labels is None:
        labels = [type(s).__name__ for s in items]
    names = items[0].channel_names if channels is None else tuple(channels)
    n_ch = len(names)

    fig, axes = plt.subplots(
        n_ch, len(items),
        figsize=(7.0 * len(items), 1.7 * n_ch),
        sharex="col", squeeze=False,
    )
    for col, sig in enumerate(items):
        flat = _flatten(sig)
        fs = sig.fs or 1.0
        end = min(start + length, flat.shape[1])
        t = np.arange(start, end) / fs
        axes[0, col].set_title(labels[col])
        for row, name in enumerate(names):
            ch = sig.channel_names.index(name)
            y = flat[ch, start:end].astype(np.float64)
            _draw_channel(axes[row, col], t, y, spline)
            if col == 0:
                axes[row, col].set_ylabel(name)
        axes[0, col].legend(loc="upper right", fontsize=8)
        axes[-1, col].set_xlabel("time [s]")

    fig.suptitle(title or f"Waveforms (start={start}, length={length})")
    fig.tight_layout()
    return fig


def _harmonic_ylim(
    values: np.ndarray,
    f_shown: np.ndarray,
    fundamental: float | None,
) -> float | None:
    """고조파가 잘 보이도록 y 상한을 계산한다.

    정규화 데이터 스펙트럼에서 가장 큰 두 성분 — 0 Hz 부근 DC 누설
    스커트와 기본파 첨두 — 은 고조파(2·3·5…차)를 수십~수백 배 압도한다.
    둘 다 y 스케일 계산에서 제외하고 '고조파 영역'(기본파의 0.5배 이상,
    기본파 ±30% 제외)의 최댓값에 30% 여유를 더해 상한으로 삼는다.
    기본파·스커트 막대는 위로 잘리고 고조파가 축을 채운다.

    `fundamental`을 모르면(파형 폴더명에 rpm 없음) None — 클리핑 생략.
    제외 후에도 압도하는 첨두가 없으면(상한이 전체 최댓값의 90% 이상)
    역시 None.
    """
    if values.size == 0 or not fundamental or fundamental <= 0.0:
        return None
    peak = float(values.max())
    if peak <= 0.0:
        return None
    # 고조파 영역: DC 스커트(기본파 0.5배 미만)와 기본파(±30%) 제외.
    eligible = (f_shown >= 0.5 * fundamental) & (
        np.abs(f_shown - fundamental) >= 0.3 * fundamental
    )
    work = np.where(eligible, values, 0.0).astype(np.float64)
    if not np.any(work > 0.0):
        return None
    # 고조파 영역 안에서도 홀로 압도하는 첨두(나머지 최댓값의 4배 초과,
    # 예: PWM 성분)는 그 클러스터를 제거하고 반복 — 대표 고조파만 남긴다.
    n = work.size
    for _ in range(6):
        pk = int(np.argmax(work))
        m = float(work[pk])
        lo = pk
        while lo > 0 and 0.0 < work[lo - 1] <= work[lo]:
            lo -= 1
        hi = pk
        while hi < n - 1 and 0.0 < work[hi + 1] <= work[hi]:
            hi += 1
        rest_max = max(
            float(work[:lo].max(initial=0.0)),
            float(work[hi + 1:].max(initial=0.0)),
        )
        if rest_max <= 0.0 or m <= 4.0 * rest_max:
            break  # 더는 홀로 압도하지 않음 — 이 m이 대표 고조파.
        work[lo:hi + 1] = 0.0
    top = float(work.max()) * 1.3
    return top if top < 0.9 * peak else None


def plot_fft_bars(
    fft,
    *,
    max_freq: float | None = None,
    channels: tuple[str, ...] | None = None,
    fundamental: float | None = None,
    clip_to_harmonics: bool = True,
    title: str | None = None,
):
    """FFTData의 채널별 진폭 스펙트럼을 막대그래프로 그린다.

    그룹 축으로 평균한 대표 스펙트럼을 채널마다 한 칸씩 그린다. DC(0 Hz)
    빈은 정규화 데이터에서 압도적으로 커 해석을 방해하므로 제외한다.
    `fundamental`을 주면 그 정수배 위치에 고조파 보조선을 긋는다(Old).

    `clip_to_harmonics`가 참이면 지배적 첨두(0 Hz 부근 DC 누설 스커트와
    기본파)에 눌리지 않도록 y 상한을 고조파 기준으로 낮춘다. 상한을
    넘는 막대는 위로 잘리고 그 실제 높이는 막대 위 라벨로 표기한다.

    빈 수가 많으면(>1500) 막대를 패치 대신 단일 LineCollection
    (`vlines`)으로 그려 렌더링 속도를 유지한다.
    """
    names = fft.channel_names if channels is None else tuple(channels)
    freqs = fft.freqs
    if max_freq is None:
        max_freq = float(freqs[-1])
    # DC 빈 제외 + 표시 상한 적용.
    mask = (freqs > 0.0) & (freqs <= max_freq)
    f_shown = freqs[mask]
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    n_ch = len(names)
    fig, axes = plt.subplots(
        n_ch, 1, figsize=(13.0, 1.9 * n_ch), sharex=True, squeeze=False,
    )
    many_bins = len(f_shown) > 1500
    for i, name in enumerate(names):
        shown = fft.channels[name].mean(axis=0)[mask]
        ax = axes[i, 0]
        color = f"C{i % 10}"
        if many_bins:
            # 빈이 많으면 막대(패치 N개)는 렌더가 매우 느리다 — 단일
            # LineCollection인 vlines로 같은 모양을 빠르게 그린다.
            ax.vlines(f_shown, 0.0, shown, color=color, linewidth=0.6)
        else:
            ax.bar(f_shown, shown, width=df, color=color,
                   edgecolor="none", align="center")
        ax.set_ylabel(f"{name}  |X|")
        ax.grid(True, alpha=0.3, axis="y")
        if fundamental:
            harm = fundamental
            while harm <= max_freq:
                ax.axvline(harm, color="k", linewidth=0.4, alpha=0.18)
                harm += fundamental
        if clip_to_harmonics:
            top = _harmonic_ylim(shown, f_shown, fundamental)
            if top is not None:
                ax.set_ylim(0.0, top)
                # 상한을 넘어 잘린 기본파 막대의 실제 높이를 라벨로.
                j = int(np.argmax(shown))
                if shown[j] > top:
                    ax.annotate(
                        f"↑ {shown[j]:.3g}",
                        xy=(f_shown[j], top), xytext=(0, -2),
                        textcoords="offset points",
                        ha="center", va="top", fontsize=7, color="k",
                    )
    axes[-1, 0].set_xlabel("frequency [Hz]")
    fig.suptitle(
        title or f"FFT amplitude spectrum (group-averaged, 0–{max_freq:.0f} Hz)"
    )
    fig.tight_layout()
    return fig


def _grid_shape(n: int) -> tuple[int, int]:
    """채널쌍 개수 n에 맞는 (행, 열) — 3열 고정 (9쌍이면 3×3)."""
    ncols = 3
    return (n + ncols - 1) // ncols, ncols


def _lag_axis(xcorr, freq_resolution: float | None) -> tuple[np.ndarray, str]:
    """xcorr 종류에 맞춰 lag 축을 물리 단위로 환산하고 축 라벨을 정한다."""
    lags = xcorr.lags
    if isinstance(xcorr, CrossCorrLog):
        if xcorr.fs:
            return lags / xcorr.fs * 1000.0, "lag [ms]"
        return lags.astype(np.float64), "lag [samples]"
    # CrossCorrFFT — lag 단위는 주파수 빈.
    if freq_resolution is not None:
        return lags * freq_resolution, "freq-shift lag [Hz]"
    return lags.astype(np.float64), "freq-shift lag [bins]"


def plot_xcorr_heatmaps(xcorr, *, freq_resolution: float | None = None,
                        title: str | None = None):
    """채널쌍 상관을 (행 × lag) 히트맵 그리드로 그린다 (Old 히트맵).

    행 축은 입력 항목/그룹 인덱스, x축은 lag. 데이터에 음수가 있으면
    0 중심 발산 컬러맵(RdBu_r), 없으면 순차 컬러맵(viridis)을 쓴다.
    """
    pairs = xcorr.pairs
    nrows, ncols = _grid_shape(len(pairs))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(15.0, 3.4 * nrows),
        constrained_layout=True, squeeze=False,
    )
    axes_flat = axes.flatten()
    lag_axis, lag_label = _lag_axis(xcorr, freq_resolution)

    all_values = np.concatenate(
        [xcorr.pair_data[f"{a}-{b}"].ravel() for a, b in pairs]
    )
    peak = float(np.abs(all_values).max()) or 1.0
    if all_values.min() < 0.0:
        # 부호 있는 상관(정규화 데이터) — 0 중심 발산 컬러맵.
        imshow_kw = {"cmap": "RdBu_r", "vmin": -peak, "vmax": peak}
        cbar_label = "cross-correlation"
    else:
        # 모두 0 이상 — lag 0 첨두가 압도적으로 커 동적 범위가 넓다.
        # 로그 색상으로 첨두 주변의 falloff와 그룹별 변화를 드러낸다.
        imshow_kw = {
            "cmap": "viridis",
            "norm": LogNorm(vmin=max(peak / 1e6, 1e-12), vmax=peak),
        }
        cbar_label = "cross-correlation (log color scale)"

    im = None
    for ax, (a, b) in zip(axes_flat, pairs):
        m = xcorr.pair_data[f"{a}-{b}"]
        im = ax.imshow(
            m, aspect="auto", origin="lower",
            extent=[lag_axis[0], lag_axis[-1], 0, m.shape[0]],
            interpolation="nearest", **imshow_kw,
        )
        ax.set_title(f"{a}-{b}")
        ax.set_xlabel(lag_label)
        ax.set_ylabel("group index")
        ax.axvline(0.0, color="k", linewidth=0.5, alpha=0.4)
    for ax in axes_flat[len(pairs):]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(
            im, ax=axes_flat.tolist(), shrink=0.85, label=cbar_label,
        )
    fig.suptitle(title or "Cross-correlation heatmaps (row × lag)")
    return fig


def _profile_ylim(profile: np.ndarray) -> tuple[float, float]:
    """프로파일이 잘 보이도록 y 범위 ``(y0, y1)``를 정한다.

    두 가지 "밋밋함"을 모두 푼다:

    * 첨두가 본문(절댓값 95퍼센타일)을 4배 넘게 압도하면, 본문 변동에
      맞춰 범위를 조인다 — 첨두는 축을 벗어나 잘린다.
    * 그렇지 않으면 데이터 실제 범위에 15% 여백만 둔다. 값이 좁은 띠에
      몰려 있어도(예: 1.0 부근) 그 띠를 확대해 잔물결을 드러낸다.
    """
    lo, hi = float(profile.min()), float(profile.max())
    absdev = np.abs(profile)
    peak = float(absdev.max())
    body = float(np.percentile(absdev, 95.0))
    if body > 0.0 and peak > 4.0 * body:
        lim = body * 1.4
        return (-lim, lim) if lo < 0.0 else (0.0, lim)
    span = hi - lo
    margin = span * 0.15 if span > 0.0 else (abs(hi) * 0.15 or 1.0)
    return lo - margin, hi + margin


def plot_xcorr_profiles(xcorr, *, freq_resolution: float | None = None,
                        title: str | None = None):
    """채널쌍 상관의 행-평균 프로파일을 선그래프 그리드로 그린다 (Old 프로파일).

    각 채널쌍 행렬을 행 축으로 평균해 lag에 대한 1D 곡선을 만든다.
    채널쌍마다 다른 색을 쓰고 곡선 아래를 채워 입체감을 준다. 첨두가
    본문을 압도하면 y 범위를 본문에 맞춰 조여 잔물결을 드러내고, 이때
    축을 벗어난 첨두는 화살표 주석으로 실제 값을 표기한다.
    """
    pairs = xcorr.pairs
    nrows, ncols = _grid_shape(len(pairs))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(15.0, 3.0 * nrows),
        sharex=True, constrained_layout=True, squeeze=False,
    )
    axes_flat = axes.flatten()
    lag_axis, lag_label = _lag_axis(xcorr, freq_resolution)

    for idx, (ax, (a, b)) in enumerate(zip(axes_flat, pairs)):
        profile = xcorr.pair_data[f"{a}-{b}"].mean(axis=0)
        color = f"C{idx % 10}"

        # y 범위를 먼저 확정 — fill_between이 자동스케일을 흔들지 않도록.
        y0, y1 = _profile_ylim(profile)
        ax.set_ylim(y0, y1)
        baseline = 0.0 if y0 <= 0.0 <= y1 else y0

        ax.fill_between(lag_axis, profile, baseline, color=color, alpha=0.25)
        ax.plot(lag_axis, profile, color=color, linewidth=1.2)
        ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.4)
        ax.axvline(0.0, color="k", linewidth=0.5, alpha=0.4)

        pk = int(np.argmax(np.abs(profile)))
        pk_x, pk_y = float(lag_axis[pk]), float(profile[pk])
        ax.axvline(pk_x, color=color, linewidth=0.8, alpha=0.5, ls="--")

        if y0 <= pk_y <= y1:
            ax.plot(pk_x, pk_y, "o", markersize=6, color=color,
                    markeredgecolor="k", markeredgewidth=0.6)
        else:
            # 첨두가 축을 벗어남 — 마커 대신 화살표 주석으로 값 표기.
            edge = y1 if pk_y > y1 else y0
            ax.annotate(
                f"peak {pk_y:+.3g}", xy=(pk_x, edge),
                xytext=(0, -16 if pk_y > y1 else 16),
                textcoords="offset points", ha="center", fontsize=8,
                color=color,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
            )

        ax.set_title(f"{a}-{b}  peak @ {pk_x:+.3g} (mean={pk_y:+.3g})")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(lag_label)
    for ax in axes_flat[len(pairs):]:
        ax.axis("off")
    fig.suptitle(title or "Time/row-averaged cross-correlation profiles")
    return fig


def save_figure(fig, path: str | Path, *, dpi: int = 120) -> None:
    """Figure를 PNG로 저장하고 메모리에서 닫는다."""
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
