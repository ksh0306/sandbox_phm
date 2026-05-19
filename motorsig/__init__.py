"""motorsig: 모터 3상 전압·전류(fast_adc) 신호 전처리/분석 라이브러리.

단계별 클래스 계층 (DesignSpec.md v2):

    SignalData
       ├── FastAdcData ── LogNormalized
       ├── FFTData
       ├── CrossCorrLog
       └── CrossCorrFFT

파이프라인::

    FastAdcData ─→ LogNormalized ─→ CrossCorrLog
    FastAdcData ─→ LogNormalized ─→ FFTData ─→ CrossCorrFFT
"""

from .base import SignalData
from .fastadc import FastAdcData, concat_fast_adc
from .fft import FFTData
from .lognorm import LogNormalized, log16_plus1, shift_to_zero_baseline
from .xcorr import CrossCorrFFT, CrossCorrLog

__all__ = [
    "SignalData",
    "FastAdcData",
    "LogNormalized",
    "FFTData",
    "CrossCorrLog",
    "CrossCorrFFT",
    "concat_fast_adc",
    "log16_plus1",
    "shift_to_zero_baseline",
]
