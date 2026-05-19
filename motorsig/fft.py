"""FFTData: 정규화 데이터의 주파수 성분 분석.

채널(va, vb, vc, ia, ib, ic)을 각각 개별 필드로 보유한다 — 클래스에서는
``channels`` 딕셔너리로, h5에서는 ``/spectrum`` 그룹 아래 채널별 데이터셋으로
저장한다 (DesignSpec.md §4).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .base import SignalData
from .fastadc import _decode_names
from .lognorm import LogNormalized


def _as_lognorm_list(source) -> list[LogNormalized]:
    """입력을 LogNormalized 리스트로 정규화. 미정규화 입력은 ValueError."""
    items = list(source) if isinstance(source, list) else [source]
    if not items:
        raise ValueError("FFTData: 빈 입력은 허용하지 않는다.")
    for item in items:
        if not isinstance(item, LogNormalized):
            raise ValueError(
                "FFTData 입력은 정규화된 LogNormalized만 허용한다 "
                f"(받은 타입: {type(item).__name__})."
            )
    head = items[0]
    for other in items[1:]:
        if other.channel_names != head.channel_names:
            raise ValueError("리스트 원소의 channel_names가 서로 다르다.")
    return items


class FFTData(SignalData):
    """패킷 묶음 단위 주파수 성분.

    저장 레이아웃: [리스트(그룹), 채널, 데이터(주파수 빈)].
    채널은 ``channels`` 딕셔너리에 채널명별 개별 필드로 보유하며, 각 필드는
    shape ``(n_groups, n_freq)``. ``data`` 프로퍼티로 [그룹,채널,데이터]
    3D 배열도 얻을 수 있다. 모든 그룹은 동일 FFT 길이로 0-패딩된다.
    """

    def __init__(
        self,
        source: LogNormalized | list[LogNormalized],
        *,
        packets_per_group: int = 0,
    ):
        items = _as_lognorm_list(source)
        head = items[0]

        # 시간영역 그룹 수집: 각 그룹 (n_channels, n_samples).
        groups: list[np.ndarray] = []
        for item in items:
            n_packets, n_ch, n_per = item.data.shape
            if packets_per_group <= 0:
                chunks = [item.data]
            else:
                chunks = [
                    item.data[i : i + packets_per_group]
                    for i in range(0, n_packets, packets_per_group)
                ]
            for chunk in chunks:
                # (gp, C, S) → (C, gp*S): 패킷·샘플 축을 이어붙임.
                groups.append(chunk.transpose(1, 0, 2).reshape(n_ch, -1))

        n_fft = max(g.shape[1] for g in groups)
        specs = []
        for g in groups:
            padded = np.zeros((g.shape[0], n_fft), dtype=np.float64)
            padded[:, : g.shape[1]] = g
            specs.append(np.abs(np.fft.rfft(padded, axis=-1)))
        stacked = np.stack(specs, axis=0)  # (n_groups, n_channels, n_freq)

        self.channel_names = head.channel_names
        # 채널별 개별 필드 — 각 (n_groups, n_freq).
        self.channels = {
            name: np.ascontiguousarray(stacked[:, i, :])
            for i, name in enumerate(self.channel_names)
        }
        self.fs = head.fs
        self.packets_per_group = int(packets_per_group)
        self.n_fft = int(n_fft)
        self.freqs = np.fft.rfftfreq(
            n_fft, d=1.0 / head.fs if head.fs else 1.0
        )

    @property
    def data(self) -> np.ndarray:
        """채널 필드를 [그룹, 채널, 데이터] 3D 배열로 쌓아 반환."""
        return np.stack(
            [self.channels[name] for name in self.channel_names], axis=1
        )

    def summary(self) -> dict:
        first = self.channels[self.channel_names[0]]
        return {
            "shape": (first.shape[0], len(self.channel_names), first.shape[1]),
            "dtype": str(first.dtype),
            "n_groups": first.shape[0],
            "n_channels": len(self.channel_names),
            "n_freq": first.shape[1],
            "channels": self.channel_names,
            "fs": self.fs,
            "packets_per_group": self.packets_per_group,
            "n_fft": self.n_fft,
        }

    def plot(self, *, channels=None, ax=None, group=0, **kw):
        """지정 그룹의 채널별 진폭 스펙트럼을 그린다."""
        import matplotlib.pyplot as plt

        names = self.channel_names if channels is None else tuple(channels)
        if ax is None:
            _, ax = plt.subplots()
        for name in names:
            ax.plot(self.freqs, self.channels[name][group], label=name, **kw)
        ax.set_xlabel("frequency (Hz)" if self.fs else "bin")
        ax.set_ylabel("amplitude")
        ax.set_title(f"{type(self).__name__} (group {group})")
        ax.legend()
        return ax

    def to_h5(self, path: str | Path) -> None:
        """채널을 /spectrum 그룹 아래 채널별 데이터셋으로 저장."""
        with h5py.File(path, "w") as f:
            f.attrs["signal_kind"] = type(self).__name__
            f.attrs["packets_per_group"] = self.packets_per_group
            f.attrs["n_fft"] = self.n_fft
            f.attrs["channel_names"] = list(self.channel_names)
            if self.fs is not None:
                f.attrs["fs"] = self.fs
            spectrum = f.create_group("spectrum")
            for name in self.channel_names:
                spectrum.create_dataset(name, data=self.channels[name])
            f.create_dataset("freqs", data=self.freqs)

    @classmethod
    def from_h5(cls, path: str | Path) -> FFTData:
        path = Path(path)
        with h5py.File(path, "r") as f:
            root = dict(f.attrs)
            channel_names = _decode_names(root["channel_names"])
            spectrum = f["spectrum"]
            channels = {name: spectrum[name][...] for name in channel_names}
            freqs = f["freqs"][...]
        obj = cls.__new__(cls)
        obj.channel_names = tuple(channel_names)
        obj.channels = channels
        obj.freqs = freqs
        obj.fs = root.get("fs")
        obj.packets_per_group = int(root.get("packets_per_group", 0))
        first = channels[channel_names[0]]
        obj.n_fft = int(root.get("n_fft", (first.shape[1] - 1) * 2))
        return obj
