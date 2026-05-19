"""LogNormalized: fast_adc 데이터의 Log16(X+1) 정규화 (FastAdcData IS-A)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .fastadc import FastAdcData, _decode_names


def log16_plus1(x: np.ndarray) -> np.ndarray:
    """``y = log16(X + 1)`` 를 오버플로 없이, 부동소수 연산을 최소화해 계산.

    명세 §3.2 구현 요구:

    (a) X+1 오버플로 방지 — X가 부호 없는 정수라 최댓값에서 ``+1``이
        0으로 래핑될 수 있다. 연산 전 uint64로 승격해 래핑을 막는다.

    (b) Log16 부동소수 연산 최소화 — ``log16(n) = log2(n) / 4`` 이고
        ``log2(n)``의 정수부는 비트 추출(``np.frexp``의 지수)로 구한다.
        가수부 보정에만 ``np.log2``를 1회 쓴다.
        즉 ``log2(n) = 정수부(exp-1) + log2(가수부)``.

    입력은 부호 없는 정수 ndarray, 출력은 float64 (범위 0 ~ bits/4).
    """
    if not np.issubdtype(np.asarray(x).dtype, np.unsignedinteger):
        raise ValueError("log16_plus1 입력은 부호 없는 정수여야 한다.")

    # (a) uint64로 승격 후 +1 — 래핑 없음.
    n = x.astype(np.uint64) + np.uint64(1)

    # (b) frexp로 n = mant * 2**exp (mant in [0.5, 1)) 분해.
    #     정수부 floor(log2(n)) = exp - 1 (지수 비트 추출, 부동소수 log 아님).
    #     가수부 보정 log2(2*mant) in [0, 1) — 부동소수 log 1회.
    mant, exp = np.frexp(n.astype(np.float64))
    log2_n = (exp - 1).astype(np.float64) + np.log2(2.0 * mant)
    return log2_n / 4.0


def shift_to_zero_baseline(
    data: np.ndarray,
    channel_names: tuple[str, ...],
) -> np.ndarray:
    """정규화 전처리: 전압군·전류군을 각각 최솟값이 0이 되도록 평행이동.

    전류 신호는 DC 오프셋(~32768) 위에 작은 진폭으로 진동해 raw 값이
    좁은 고값 구간(예: 31000~34000)에 몰린다. Log16 정규화는 큰 값
    구간을 심하게 압축하므로, 이 상태로 정규화하면 출력이 좁은 띠에
    뭉친다. 정규화 전에 데이터를 0 근처로 끌어내리면 같은 진폭이 훨씬
    넓은 log 비율 구간에 펼쳐져 고른 분포를 얻는다.

    - 전압 채널(v*) 묶음과 전류 채널(i*) 묶음을 각각 독립으로 처리해,
      각 묶음의 최솟값이 0이 되도록 평행이동한다. 묶음 단위로 같은
      양만큼 옮기므로 3상 간 진폭·위상 차이는 보존된다.
    - 최솟값을 빼므로 결과는 항상 0 이상 — 음수 클램프가 필요 없다.

    원본 ``data``는 변경하지 않고 새 배열을 반환한다. 반환 dtype은 부호
    없는 정수(uint64)로, 곧바로 :func:`log16_plus1`에 넣을 수 있다.
    """
    names = [str(n).lower() for n in channel_names]
    v_idx = [i for i, n in enumerate(names) if n.startswith("v")]
    i_idx = [i for i, n in enumerate(names) if n.startswith("i")]
    if not v_idx or not i_idx:
        raise ValueError(
            "채널명에서 전압(v*)·전류(i*) 채널을 모두 식별해야 한다."
        )

    # int64로 승격해 복사 — 원본 보존 + 빼기 중 부호없는 정수 언더플로 방지.
    work = np.asarray(data).astype(np.int64)
    work[:, v_idx, :] -= work[:, v_idx, :].min()  # 전압군 최솟값 → 0.
    work[:, i_idx, :] -= work[:, i_idx, :].min()  # 전류군 최솟값 → 0.
    return work.astype(np.uint64)


class LogNormalized(FastAdcData):
    """fast_adc 데이터를 Log16(X+1)로 정규화한 결과.

    FastAdcData를 IS-A 상속하여 저장/읽기/시각화 인터페이스를 그대로
    재사용한다. 단 data dtype은 float64이며, 출력 범위는 0 ~ bits/4.
    """

    def __init__(self, source_data: FastAdcData):
        if not isinstance(source_data, FastAdcData):
            raise TypeError("LogNormalized 입력은 FastAdcData여야 한다.")
        if isinstance(source_data, LogNormalized):
            raise TypeError("이미 정규화된 데이터를 다시 정규화할 수 없다.")
        # 정규화 전처리: 전압군·전류군을 각각 최솟값이 0이 되도록 끌어내려
        # Log16 압축으로 값이 뭉치지 않게 한 뒤 정규화. shift_to_zero_baseline
        # / log16_plus1 모두 새 배열을 반환하므로 원본은 변경되지 않는다.
        prepped = shift_to_zero_baseline(
            source_data.data, source_data.channel_names
        )
        self.data = log16_plus1(prepped)
        self.channel_names = source_data.channel_names
        self.bits = source_data.bits
        self.fs = source_data.fs
        self.source = source_data.source

    @classmethod
    def _from_fields(cls, data, channel_names, bits, fs, source) -> LogNormalized:
        """정규화 재계산 없이 필드로부터 인스턴스 복원 (from_h5 용)."""
        obj = cls.__new__(cls)
        obj.data = np.asarray(data, dtype=np.float64)
        obj.channel_names = tuple(channel_names)
        obj.bits = int(bits)
        obj.fs = None if fs is None else float(fs)
        obj.source = source
        return obj

    def summary(self) -> dict:
        info = super().summary()
        info["value_range"] = (float(self.data.min()), float(self.data.max()))
        info["nominal_max"] = self.bits / 4
        return info

    @classmethod
    def from_h5(cls, path: str | Path) -> LogNormalized:
        """to_h5로 저장한 정규화 데이터를 읽어 복원."""
        path = Path(path)
        with h5py.File(path, "r") as f:
            if "fast_adc" not in f:
                raise ValueError(f"{path}에 fast_adc 데이터셋이 없다.")
            dset = f["fast_adc"]
            data = dset[...]
            attrs = dict(dset.attrs)
        channel_names = _decode_names(attrs["channel_names"])
        return cls._from_fields(
            data,
            channel_names,
            attrs["bits"],
            attrs.get("fs"),
            attrs.get("source"),
        )
