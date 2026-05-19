"""motorsig CLI — 단계 클래스를 조합만 하는 얇은 래퍼 (수치 로직 없음)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .fastadc import FastAdcData, concat_fast_adc
from .fft import FFTData
from .lognorm import LogNormalized
from .xcorr import CrossCorrFFT, CrossCorrLog


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motorsig",
        description="fast_adc h5 → 정규화 / FFT / 상관 분석 파이프라인.",
    )
    parser.add_argument("inputs", nargs="+", help="입력 h5 파일 (fast_adc 포함)")
    parser.add_argument(
        "--analysis",
        choices=["fft", "xcorr-log", "xcorr-fft"],
        default="fft",
        help="수행할 분석 단계 (기본: fft)",
    )
    parser.add_argument(
        "--packets-per-group",
        type=int,
        default=0,
        help="FFT 그룹당 패킷 수 (0=전체 1그룹)",
    )
    parser.add_argument(
        "--max-lag", type=int, default=None, help="xcorr-log 최대 lag"
    )
    parser.add_argument("--out", required=True, help="결과 h5 저장 경로")
    return parser


def main(argv: list[str] | None = None) -> None:
    """파일 적재 → 결합 → 정규화 → 분석 → 저장. 조합만 수행한다."""
    args = _build_parser().parse_args(argv)

    raw = concat_fast_adc([FastAdcData.from_h5(p) for p in args.inputs])
    norm = LogNormalized(raw)

    if args.analysis == "fft":
        result = FFTData(norm, packets_per_group=args.packets_per_group)
    elif args.analysis == "xcorr-log":
        result = CrossCorrLog(norm, max_lag=args.max_lag)
    else:  # xcorr-fft
        fft = FFTData(norm, packets_per_group=args.packets_per_group)
        result = CrossCorrFFT(fft)

    out = Path(args.out)
    result.to_h5(out)
    result.describe()
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
