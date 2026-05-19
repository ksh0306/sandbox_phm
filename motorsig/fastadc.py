"""FastAdcData: 원본 fast_adc 데이터와 다중 파일 결합 함수."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .base import SignalData

# 부호 있는 정수 dtype → 같은 폭의 부호 없는 dtype.
_SIGNED_TO_UNSIGNED = {
    np.dtype(np.int8): np.uint8,
    np.dtype(np.int16): np.uint16,
    np.dtype(np.int32): np.uint32,
    np.dtype(np.int64): np.uint64,
}


def _decode_names(raw) -> tuple[str, ...]:
    """h5 attr로 저장된 채널명을 문자열 튜플로 복원."""
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode()
    if isinstance(raw, str):
        return tuple(name.strip() for name in raw.split(","))
    return tuple(
        item.decode() if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in raw
    )


class FastAdcData(SignalData):
    """ADC 모듈로 들어온 부호 없는 정수 fast_adc 데이터.

    data:  np.ndarray, shape = (n_packets, n_channels, n_per_packet)
           dtype = 부호 없는 정수 (uint16 등). 원본 보존, 캐스팅 금지.
    channel_names: ("v1","v2","v3","i1","i2","i3") 등 순서 고정
    bits:  ADC 비트폭 (정규화에서 사용)
    fs:    샘플링 주파수 (Hz). 알면 기록, 몰라도 됨
    source: 출처 문자열
    """

    def __init__(self, data, channel_names, bits, *, fs=None, source=None):
        data = np.asarray(data)
        channel_names = tuple(channel_names)
        self._validate(data, channel_names)
        self.data = data
        self.channel_names = channel_names
        self.bits = int(bits)
        self.fs = None if fs is None else float(fs)
        self.source = source

    @staticmethod
    def _validate(data: np.ndarray, channel_names: tuple[str, ...]) -> None:
        """축·채널 수·dtype 계약을 검사. 위반 시 ValueError."""
        if data.ndim != 3:
            raise ValueError(
                f"data.ndim은 3이어야 한다 (받은 값: {data.ndim})."
            )
        if data.shape[1] != len(channel_names):
            raise ValueError(
                f"채널 축 크기 {data.shape[1]}와 channel_names 길이 "
                f"{len(channel_names)}가 일치하지 않는다."
            )
        if not np.issubdtype(data.dtype, np.unsignedinteger):
            raise ValueError(
                f"data dtype은 부호 없는 정수여야 한다 (받은 값: {data.dtype})."
            )

    # ── 데이터 확인 ──
    def summary(self) -> dict:
        return {
            "shape": self.data.shape,
            "dtype": str(self.data.dtype),
            "n_packets": self.data.shape[0],
            "n_channels": self.data.shape[1],
            "n_per_packet": self.data.shape[2],
            "channel_names": self.channel_names,
            "bits": self.bits,
            "fs": self.fs,
            "source": self.source,
        }

    # ── 시각화 ──
    def plot(self, *, channels=None, ax=None, **kw):
        """채널별 시계열(패킷을 이어붙인 연속 신호)을 그린다."""
        import matplotlib.pyplot as plt

        names = self.channel_names if channels is None else tuple(channels)
        if ax is None:
            _, ax = plt.subplots()
        flat = self.data.reshape(
            self.data.shape[0], self.data.shape[1], -1
        ).transpose(1, 0, 2)
        flat = flat.reshape(self.data.shape[1], -1)
        for name in names:
            idx = self.channel_names.index(name)
            ax.plot(flat[idx], label=name, **kw)
        ax.set_xlabel("sample")
        ax.set_ylabel("ADC code")
        ax.set_title(type(self).__name__)
        ax.legend()
        return ax

    # ── 저장 / 읽기 ──
    def to_h5(self, path: str | Path) -> None:
        """파일명을 지정하여 저장. 레이아웃 [패킷, 채널, 패킷당데이터]."""
        with h5py.File(path, "w") as f:
            f.attrs["signal_kind"] = type(self).__name__
            dset = f.create_dataset("fast_adc", data=self.data)
            dset.attrs["channel_names"] = list(self.channel_names)
            dset.attrs["bits"] = self.bits
            if self.fs is not None:
                dset.attrs["fs"] = self.fs
            if self.source is not None:
                dset.attrs["source"] = str(self.source)

    @classmethod
    def from_h5(cls, path: str | Path) -> FastAdcData:
        """Motor** 폴더의 h5 포맷 fast_adc 데이터를 읽어온다.

        명세 §2.2 참조: 원본 파일의 ``fast_adc``는 int16로 저장돼 있으나
        실제 ADC 코드는 부호 없는 16bit 값이다. 같은 비트패턴을 유지한 채
        ``view``로 부호 없는 dtype으로 재해석한다(값 변경 없음).
        """
        path = Path(path)
        with h5py.File(path, "r") as f:
            if "fast_adc" not in f:
                raise ValueError(
                    f"{path}에 fast_adc 데이터셋이 없다 (파생 파일일 수 있음)."
                )
            dset = f["fast_adc"]
            raw = dset[...]
            attrs = dict(dset.attrs)
            root = dict(f.attrs)

        # 채널명: 우리 포맷(channel_names) 또는 원본 포맷(channel_order).
        if "channel_names" in attrs:
            channel_names = _decode_names(attrs["channel_names"])
        elif "channel_order" in attrs:
            channel_names = _decode_names(attrs["channel_order"])
        else:
            channel_names = tuple(f"ch{i}" for i in range(raw.shape[1]))

        bits = attrs.get("bits", root.get("adc_effective_bits"))
        if bits is None:
            raise ValueError(f"{path}에서 ADC 비트폭을 찾을 수 없다.")
        fs = attrs.get("fs", root.get("fs_hz"))
        source = attrs.get("source", path.name)

        # int16 등 부호 있는 저장 → 동일 비트패턴의 부호 없는 dtype 재해석.
        if np.issubdtype(raw.dtype, np.signedinteger):
            raw = raw.view(_SIGNED_TO_UNSIGNED[raw.dtype])

        return cls(raw, channel_names, int(bits), fs=fs, source=source)


def concat_fast_adc(items: list[FastAdcData]) -> FastAdcData:
    """동일 포맷의 여러 FastAdcData를 패킷 축(axis=0)으로 연결.

    channel_names/bits/dtype/패킷당샘플 수 불일치 시 ValueError.
    """
    if not items:
        raise ValueError("concat_fast_adc: 빈 리스트는 결합할 수 없다.")

    head = items[0]
    for other in items[1:]:
        if other.channel_names != head.channel_names:
            raise ValueError("channel_names가 서로 다르다.")
        if other.bits != head.bits:
            raise ValueError("bits가 서로 다르다.")
        if other.data.dtype != head.data.dtype:
            raise ValueError("data dtype이 서로 다르다.")
        if other.data.shape[1:] != head.data.shape[1:]:
            raise ValueError("채널 수 또는 패킷당 샘플 수가 서로 다르다.")

    merged = np.concatenate([item.data for item in items], axis=0)
    sources = [item.source for item in items if item.source is not None]
    return FastAdcData(
        merged,
        head.channel_names,
        head.bits,
        fs=head.fs,
        source="+".join(str(s) for s in sources) if sources else None,
    )
