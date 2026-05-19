"""CrossCorrLog / CrossCorrFFT: 채널쌍 상관 분석.

채널쌍을 각각 개별 필드로 보유한다 — 클래스에서는 ``pair_data``
딕셔너리("a-b" 키)로, h5에서는 ``/xcorr`` 그룹 아래 쌍별 데이터셋으로 저장한다.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import h5py
import numpy as np
from scipy import signal

from .base import SignalData
from .fastadc import _decode_names
from .fft import FFTData
from .lognorm import LogNormalized


def _resolve_pairs(channel_names, pairs):
    """pairs(None=전체 조합)를 (이름쌍, 인덱스쌍) 리스트로 해석."""
    if pairs is None:
        name_pairs = list(itertools.combinations(channel_names, 2))
    else:
        name_pairs = [tuple(p) for p in pairs]
    index = {name: i for i, name in enumerate(channel_names)}
    idx_pairs = []
    for a, b in name_pairs:
        if a not in index or b not in index:
            raise ValueError(f"채널쌍 ({a}, {b})에 알 수 없는 채널명이 있다.")
        idx_pairs.append((index[a], index[b]))
    return name_pairs, idx_pairs


def _xcorr(a: np.ndarray, b: np.ndarray, normalize: bool) -> np.ndarray:
    """a, b의 전체(full) 상호상관. normalize 시 에너지로 정규화."""
    full = signal.correlate(a, b, mode="full")
    if normalize:
        denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
        if denom > 0:
            full = full / denom
    return full


def _decode_names_raw(raw) -> list[str]:
    """문자열 리스트 attr를 파이썬 str 리스트로 복원 (분할 없음)."""
    return [
        item.decode() if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in raw
    ]


def _pair_key(a: str, b: str) -> str:
    """채널쌍 (a, b)의 필드/데이터셋 키."""
    return f"{a}-{b}"


class CrossCorrLog(SignalData):
    """Log 정규화 데이터 간 채널쌍 상관 (시간영역).

    저장 레이아웃: [입력항목, 채널쌍, lag]. 채널쌍은 ``pair_data``
    딕셔너리("a-b" 키)에 쌍별 개별 필드(각 shape ``(n_items, n_lags)``)로
    보유한다. ``data`` 프로퍼티로 3D 배열도 얻을 수 있다.

    pair (A, B)에 대해 ``lag = argmax``는 A가 B보다 lag 샘플만큼
    지연됐음을 뜻한다(``scipy.signal.correlation_lags`` 규약).
    """

    def __init__(
        self,
        source: LogNormalized | list[LogNormalized],
        *,
        pairs: list[tuple[str, str]] | None = None,
        max_lag: int | None = None,
        normalize: bool = True,
    ):
        items = list(source) if isinstance(source, list) else [source]
        if not items:
            raise ValueError("CrossCorrLog: 빈 입력은 허용하지 않는다.")
        for item in items:
            if not isinstance(item, LogNormalized):
                raise ValueError(
                    "CrossCorrLog 입력은 LogNormalized만 허용한다 "
                    f"(받은 타입: {type(item).__name__})."
                )
        head = items[0]
        name_pairs, idx_pairs = _resolve_pairs(head.channel_names, pairs)

        # 항목별 (채널, 샘플) 시계열 — 패킷·샘플 축을 이어붙임.
        series = [
            item.data.transpose(1, 0, 2).reshape(item.data.shape[1], -1)
            for item in items
        ]
        if max_lag is None:
            eff_max_lag = min(s.shape[1] - 1 for s in series)
        else:
            eff_max_lag = int(max_lag)
        lags = np.arange(-eff_max_lag, eff_max_lag + 1)

        result = np.empty((len(items), len(idx_pairs), lags.size))
        for i, ts in enumerate(series):
            full_lags = signal.correlation_lags(
                ts.shape[1], ts.shape[1], mode="full"
            )
            keep = np.abs(full_lags) <= eff_max_lag
            for j, (ia, ib) in enumerate(idx_pairs):
                full = _xcorr(ts[ia], ts[ib], normalize)
                result[i, j] = full[keep]

        self.lags = lags
        self.pairs = name_pairs
        self.channel_names = head.channel_names
        self.fs = head.fs
        self.normalize = bool(normalize)
        # 채널쌍별 개별 필드 — 각 (n_items, n_lags).
        self.pair_data = {
            _pair_key(a, b): np.ascontiguousarray(result[:, j, :])
            for j, (a, b) in enumerate(name_pairs)
        }

    @property
    def data(self) -> np.ndarray:
        """채널쌍 필드를 [항목, 채널쌍, lag] 3D 배열로 쌓아 반환."""
        return np.stack(
            [self.pair_data[_pair_key(a, b)] for a, b in self.pairs], axis=1
        )

    def summary(self) -> dict:
        first = self.pair_data[_pair_key(*self.pairs[0])]
        return {
            "shape": (first.shape[0], len(self.pairs), first.shape[1]),
            "dtype": str(first.dtype),
            "n_items": first.shape[0],
            "n_pairs": len(self.pairs),
            "n_lags": first.shape[1],
            "pairs": self.pairs,
            "lag_range": (int(self.lags[0]), int(self.lags[-1])),
            "normalize": self.normalize,
            "fs": self.fs,
        }

    def plot(self, *, channels=None, ax=None, item=0, **kw):
        """지정 항목의 채널쌍별 상관 곡선을 그린다."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        for a, b in self.pairs:
            ax.plot(
                self.lags,
                self.pair_data[_pair_key(a, b)][item],
                label=f"{a}-{b}",
                **kw,
            )
        ax.set_xlabel("lag (samples)")
        ax.set_ylabel("correlation")
        ax.set_title(f"{type(self).__name__} (item {item})")
        ax.legend()
        return ax

    def to_h5(self, path: str | Path) -> None:
        """채널쌍을 /xcorr 그룹 아래 쌍별 데이터셋으로 저장."""
        with h5py.File(path, "w") as f:
            f.attrs["signal_kind"] = type(self).__name__
            f.attrs["normalize"] = self.normalize
            if self.fs is not None:
                f.attrs["fs"] = self.fs
            f.attrs["channel_names"] = list(self.channel_names)
            f.attrs["pairs"] = [f"{a}|{b}" for a, b in self.pairs]
            xcorr = f.create_group("xcorr")
            for a, b in self.pairs:
                xcorr.create_dataset(
                    _pair_key(a, b), data=self.pair_data[_pair_key(a, b)]
                )
            f.create_dataset("lags", data=self.lags)

    @classmethod
    def from_h5(cls, path: str | Path) -> CrossCorrLog:
        path = Path(path)
        with h5py.File(path, "r") as f:
            root = dict(f.attrs)
            pairs = [
                tuple(p.split("|")) for p in _decode_names_raw(root["pairs"])
            ]
            xcorr = f["xcorr"]
            pair_data = {
                _pair_key(a, b): xcorr[_pair_key(a, b)][...] for a, b in pairs
            }
            lags = f["lags"][...]
        obj = cls.__new__(cls)
        obj.lags = lags
        obj.pairs = pairs
        obj.pair_data = pair_data
        obj.channel_names = _decode_names(root["channel_names"])
        obj.fs = root.get("fs")
        obj.normalize = bool(root.get("normalize", True))
        return obj


class CrossCorrFFT(SignalData):
    """FFT 데이터 간 채널쌍 상관 (주파수영역).

    저장 레이아웃: [그룹, 채널쌍, lag(주파수 빈)]. 채널쌍은 ``pair_data``
    딕셔너리("a-b" 키)에 쌍별 개별 필드로 보유한다.
    """

    def __init__(
        self,
        source: FFTData | list[FFTData],
        *,
        pairs: list[tuple[str, str]] | None = None,
    ):
        items = list(source) if isinstance(source, list) else [source]
        if not items:
            raise ValueError("CrossCorrFFT: 빈 입력은 허용하지 않는다.")
        for item in items:
            if not isinstance(item, FFTData):
                raise ValueError(
                    "CrossCorrFFT 입력은 FFTData만 허용한다 "
                    f"(받은 타입: {type(item).__name__})."
                )
        head = items[0]
        n_freq = head.data.shape[2]
        for other in items[1:]:
            if other.channel_names != head.channel_names:
                raise ValueError("FFTData 항목의 channel_names가 서로 다르다.")
            if other.data.shape[2] != n_freq:
                raise ValueError("FFTData 항목의 주파수 빈 수가 서로 다르다.")
        name_pairs, idx_pairs = _resolve_pairs(head.channel_names, pairs)

        # 모든 항목의 그룹을 이어붙여 그룹 축으로 누적.
        spectra = np.concatenate([item.data for item in items], axis=0)
        lags = signal.correlation_lags(n_freq, n_freq, mode="full")

        result = np.empty((spectra.shape[0], len(idx_pairs), lags.size))
        for g in range(spectra.shape[0]):
            for j, (ia, ib) in enumerate(idx_pairs):
                result[g, j] = _xcorr(spectra[g, ia], spectra[g, ib], False)

        self.lags = lags
        self.pairs = name_pairs
        self.channel_names = head.channel_names
        self.fs = head.fs
        # 채널쌍별 개별 필드 — 각 (n_groups, n_lags).
        self.pair_data = {
            _pair_key(a, b): np.ascontiguousarray(result[:, j, :])
            for j, (a, b) in enumerate(name_pairs)
        }

    @property
    def data(self) -> np.ndarray:
        """채널쌍 필드를 [그룹, 채널쌍, lag] 3D 배열로 쌓아 반환."""
        return np.stack(
            [self.pair_data[_pair_key(a, b)] for a, b in self.pairs], axis=1
        )

    def summary(self) -> dict:
        first = self.pair_data[_pair_key(*self.pairs[0])]
        return {
            "shape": (first.shape[0], len(self.pairs), first.shape[1]),
            "dtype": str(first.dtype),
            "n_groups": first.shape[0],
            "n_pairs": len(self.pairs),
            "n_lags": first.shape[1],
            "pairs": self.pairs,
            "lag_range": (int(self.lags[0]), int(self.lags[-1])),
            "fs": self.fs,
        }

    def plot(self, *, channels=None, ax=None, group=0, **kw):
        """지정 그룹의 채널쌍별 주파수영역 상관 곡선을 그린다."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        for a, b in self.pairs:
            ax.plot(
                self.lags,
                self.pair_data[_pair_key(a, b)][group],
                label=f"{a}-{b}",
                **kw,
            )
        ax.set_xlabel("lag (frequency bins)")
        ax.set_ylabel("correlation")
        ax.set_title(f"{type(self).__name__} (group {group})")
        ax.legend()
        return ax

    def to_h5(self, path: str | Path) -> None:
        """채널쌍을 /xcorr 그룹 아래 쌍별 데이터셋으로 저장."""
        with h5py.File(path, "w") as f:
            f.attrs["signal_kind"] = type(self).__name__
            if self.fs is not None:
                f.attrs["fs"] = self.fs
            f.attrs["channel_names"] = list(self.channel_names)
            f.attrs["pairs"] = [f"{a}|{b}" for a, b in self.pairs]
            xcorr = f.create_group("xcorr")
            for a, b in self.pairs:
                xcorr.create_dataset(
                    _pair_key(a, b), data=self.pair_data[_pair_key(a, b)]
                )
            f.create_dataset("lags", data=self.lags)

    @classmethod
    def from_h5(cls, path: str | Path) -> CrossCorrFFT:
        path = Path(path)
        with h5py.File(path, "r") as f:
            root = dict(f.attrs)
            pairs = [
                tuple(p.split("|")) for p in _decode_names_raw(root["pairs"])
            ]
            xcorr = f["xcorr"]
            pair_data = {
                _pair_key(a, b): xcorr[_pair_key(a, b)][...] for a, b in pairs
            }
            lags = f["lags"][...]
        obj = cls.__new__(cls)
        obj.lags = lags
        obj.pairs = pairs
        obj.pair_data = pair_data
        obj.channel_names = _decode_names(root["channel_names"])
        obj.fs = root.get("fs")
        return obj
