"""example.py가 만든 파이프라인 h5를 읽어 각 Motor** 폴더를 시각화한다.

폴더마다 `pics/<folder>/` 아래에 PNG 5장을 생성한다:

    <folder>_waveform.png          raw vs log 정규화 6채널 시간영역 파형
    <folder>_fft_bars.png          채널별 진폭 스펙트럼 막대그래프
    <folder>_xcorr_log_profile.png 시간영역 채널쌍 상관 프로파일 (9쌍)
    <folder>_xcorr_fft_heatmap.png 주파수영역 채널쌍 상관 히트맵 (그룹×lag)
    <folder>_xcorr_fft_profile.png 주파수영역 채널쌍 상관 프로파일 (9쌍)

먼저 `python example.py`로 파이프라인 h5를 생성해 두어야 한다.
"""

from __future__ import annotations

import re
from pathlib import Path

from motorsig import CrossCorrFFT, CrossCorrLog, FastAdcData, FFTData, LogNormalized
from motorsig.visualize import (
    plot_fft_bars,
    plot_waveforms,
    plot_xcorr_heatmaps,
    plot_xcorr_profiles,
    save_figure,
)

# 모터 극쌍 수 (h5 루트 attr `pole_pairs`에서 확인한 고정값).
POLE_PAIRS = 7


def electrical_fundamental(folder_name: str) -> float | None:
    """`Motor<rpm>_*` 폴더명에서 전기 기본 주파수[Hz]를 추정.

    전기 주파수 = (rpm / 60) × 극쌍 수. 추정 불가 시 None.
    """
    match = re.match(r"Motor(\d+)", folder_name)
    if not match:
        return None
    rpm = int(match.group(1))
    return rpm / 60.0 * POLE_PAIRS


def render_folder(folder: Path, out_dir: Path) -> None:
    """한 폴더의 파이프라인 h5를 읽어 PNG 5장을 생성·저장."""
    name = folder.name
    print(f"\n=== {name} ===")
    paths = {
        "raw": folder / f"{name}_raw.h5",
        "lognorm": folder / f"{name}_lognorm.h5",
        "fft": folder / f"{name}_fft.h5",
        "xcorr_log": folder / f"{name}_xcorr_log.h5",
        "xcorr_fft": folder / f"{name}_xcorr_fft.h5",
    }
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        print(f"  파이프라인 h5 누락 {missing} — example.py를 먼저 실행하라.")
        return

    raw = FastAdcData.from_h5(paths["raw"])
    norm = LogNormalized.from_h5(paths["lognorm"])
    fft = FFTData.from_h5(paths["fft"])
    xcorr_log = CrossCorrLog.from_h5(paths["xcorr_log"])
    xcorr_fft = CrossCorrFFT.from_h5(paths["xcorr_fft"])

    out_dir.mkdir(parents=True, exist_ok=True)
    fundamental = electrical_fundamental(name)
    freq_res = float(fft.freqs[1] - fft.freqs[0])
    max_freq = (
        min(fundamental * 12.0, float(fft.freqs[-1]))
        if fundamental
        else min(2000.0, float(fft.freqs[-1]))
    )

    # 1) 시간영역 파형 (raw vs log 정규화)
    #    plot 텍스트는 영문 — matplotlib 기본 폰트에 한글 글리프가 없다.
    fig = plot_waveforms(
        [raw, norm],
        labels=["raw ADC", "log16-normalized"],
        title=f"[{name}] time-domain waveforms",
    )
    save_figure(fig, out_dir / f"{name}_waveform.png")

    # 2) FFT 진폭 스펙트럼 막대그래프
    fig = plot_fft_bars(
        fft, max_freq=max_freq, fundamental=fundamental,
        title=f"[{name}] FFT amplitude spectrum"
        + (f" (fundamental {fundamental:.0f} Hz)" if fundamental else ""),
    )
    save_figure(fig, out_dir / f"{name}_fft_bars.png")

    # 3) 시간영역 채널쌍 상관 프로파일 (CrossCorrLog는 항목 1개 → 프로파일만)
    fig = plot_xcorr_profiles(
        xcorr_log,
        title=f"[{name}] time-domain cross-correlation profiles",
    )
    save_figure(fig, out_dir / f"{name}_xcorr_log_profile.png")

    # 4) 주파수영역 채널쌍 상관 히트맵 (그룹 × 주파수 lag)
    fig = plot_xcorr_heatmaps(
        xcorr_fft, freq_resolution=freq_res,
        title=f"[{name}] frequency-domain cross-correlation heatmaps",
    )
    save_figure(fig, out_dir / f"{name}_xcorr_fft_heatmap.png")

    # 5) 주파수영역 채널쌍 상관 프로파일
    fig = plot_xcorr_profiles(
        xcorr_fft, freq_resolution=freq_res,
        title=f"[{name}] frequency-domain cross-correlation profiles",
    )
    save_figure(fig, out_dir / f"{name}_xcorr_fft_profile.png")

    for png in sorted(out_dir.glob(f"{name}_*.png")):
        print(f"  저장: {png}")


def main() -> None:
    root = Path(__file__).resolve().parent
    out_root = root / "pics"
    folders = sorted(p for p in root.glob("Motor*") if p.is_dir())
    if not folders:
        print("Motor** 폴더를 찾지 못했다.")
        return
    for folder in folders:
        render_folder(folder, out_root / folder.name)
    print(f"\n완료. PNG는 {out_root}/ 아래에 있다.")


if __name__ == "__main__":
    main()
