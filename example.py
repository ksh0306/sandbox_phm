"""각 Motor** 폴더의 raw h5를 결합하고 motorsig 파이프라인을 적용한다.

폴더마다 다음 파일을 생성한다:

    <folder>/<folder>_raw.h5         결합한 원본 FastAdcData
    <folder>/<folder>_lognorm.h5     Log16(X+1) 정규화 LogNormalized
    <folder>/<folder>_fft.h5         주파수 성분 FFTData (채널별 개별 필드)
    <folder>/<folder>_xcorr_log.h5   시간영역 채널쌍 상관 CrossCorrLog
    <folder>/<folder>_xcorr_fft.h5   주파수영역 채널쌍 상관 CrossCorrFFT

파이프라인 (DesignSpec.md §6)::

    FastAdcData ─→ LogNormalized ─→ CrossCorrLog
    FastAdcData ─→ LogNormalized ─→ FFTData ─→ CrossCorrFFT
"""

from __future__ import annotations

from pathlib import Path

from motorsig import (
    CrossCorrFFT,
    CrossCorrLog,
    FastAdcData,
    FFTData,
    LogNormalized,
    concat_fast_adc,
)

# FFT 그룹당 패킷 수 — 1600패킷 × 50샘플 = 80000샘플 = 4초 @ 20kHz.
FFT_PACKETS_PER_GROUP = 1600
# 시간영역 상관 최대 lag 기준 패킷 수 — 16000패킷(4초). 샘플 단위로 환산해 사용.
XCORR_MAX_LAG_PACKETS = 3200
# 채널쌍 (인덱스 기준): v1-v2, v2-v3, v3-v1 / i1-i2, i2-i3, i3-i1 / v1-i1, v2-i2, v3-i3.
XCORR_PAIR_INDICES = [
    (0, 1),
    (1, 2),
    (2, 0),
    (3, 4),
    (4, 5),
    (5, 3),
    (0, 3),
    (1, 4),
    (2, 5),
]


def load_raw_files(folder: Path) -> list[FastAdcData]:
    """폴더 내 fast_adc를 가진 원본 h5만 골라 FastAdcData로 적재.

    이 스크립트가 만든 결과 파일(폴더명으로 시작)은 건너뛰고, 파생 파일
    (fast_adc 없음)은 ValueError로 걸러진다.
    """
    items: list[FastAdcData] = []
    for path in sorted(folder.glob("*.h5")):
        if path.name.startswith(folder.name):
            continue  # 이 스크립트가 생성한 결과 파일
        try:
            items.append(FastAdcData.from_h5(path))
        except ValueError as exc:
            print(f"  건너뜀: {path.name} ({exc})")
    return items


def process_folder(folder: Path) -> None:
    """한 폴더의 raw 파일을 결합하고 전체 파이프라인을 적용·저장."""
    print(f"\n=== {folder.name} ===")
    raws = load_raw_files(folder)
    if not raws:
        print("  처리할 raw 파일이 없다.")
        return
    print(f"  raw 파일 {len(raws)}개 적재")

    # 1) 결합한 원본 FastAdcData
    raw = concat_fast_adc(raws)
    raw_path = folder / f"{folder.name}_raw.h5"
    raw.to_h5(raw_path)
    print(f"  FastAdcData   {raw.data.shape} -> {raw_path.name}")

    # 2) Log16(X+1) 정규화
    norm = LogNormalized(raw)
    norm_path = folder / f"{folder.name}_lognorm.h5"
    norm.to_h5(norm_path)
    print(f"  LogNormalized {norm.data.shape} -> {norm_path.name}")

    # 채널쌍·최대 lag을 실제 채널명/패킷당 샘플 수에서 산출.
    names = raw.channel_names
    pairs = [(names[i], names[j]) for i, j in XCORR_PAIR_INDICES]
    samples_per_packet = raw.data.shape[2]
    max_lag = XCORR_MAX_LAG_PACKETS * samples_per_packet

    # 3) 주파수 성분 (채널별 개별 필드)
    fft = FFTData(norm, packets_per_group=FFT_PACKETS_PER_GROUP)
    fft_path = folder / f"{folder.name}_fft.h5"
    fft.to_h5(fft_path)
    print(f"  FFTData       {fft.data.shape} -> {fft_path.name}")

    # 4) 시간영역 채널쌍 상관 (지정 9쌍, 개별 필드)
    xcorr_log = CrossCorrLog(norm, pairs=pairs, max_lag=max_lag)
    xcorr_log_path = folder / f"{folder.name}_xcorr_log.h5"
    xcorr_log.to_h5(xcorr_log_path)
    print(f"  CrossCorrLog  {xcorr_log.data.shape} -> {xcorr_log_path.name}")

    # 5) 주파수영역 채널쌍 상관 (지정 9쌍, 개별 필드)
    xcorr_fft = CrossCorrFFT(fft, pairs=pairs)
    xcorr_fft_path = folder / f"{folder.name}_xcorr_fft.h5"
    xcorr_fft.to_h5(xcorr_fft_path)
    print(f"  CrossCorrFFT  {xcorr_fft.data.shape} -> {xcorr_fft_path.name}")


def main() -> None:
    root = Path(__file__).resolve().parent
    folders = sorted(p for p in root.glob("Motor*") if p.is_dir())
    if not folders:
        print("Motor** 폴더를 찾지 못했다.")
        return
    for folder in folders:
        process_folder(folder)
    print("\n완료.")


if __name__ == "__main__":
    main()
