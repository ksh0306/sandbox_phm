from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.signal import correlate

# ─────────────────────────────────────────────────────────────────────────────
# PHM 도메인 상수
# ─────────────────────────────────────────────────────────────────────────────
CHANNEL_NAMES = ["v1", "v2", "v3", "i1", "i2", "i3"]

# v-v 위상 / i-i 위상 / v-i (역률) 9쌍.
XCORR_PAIRS = [
    ("v1", "v2"),
    ("v2", "v3"),
    ("v3", "v1"),
    ("i1", "i2"),
    ("i2", "i3"),
    ("i3", "i1"),
    ("v1", "i1"),
    ("v2", "i2"),
    ("v3", "i3"),
]

FS_HZ = 20_000
WINDOW_MS = 50
SAMPLES_PER_WINDOW = FS_HZ * WINDOW_MS // 5000  # 200 샘플 = 10 ms @ 20 kHz
MAX_LAG_SAMPLES = SAMPLES_PER_WINDOW // 2  # ±5 ms
FFT_WINDOW_SAMPLES = 100_000  # 5 초 청크
FFT_MAX_LAG_BINS = 500  # ±100 Hz (0.2 Hz/bin × 500)

_NAME_RE = re.compile(r"^([^_]+_[^_]+_[^_]+)_.+\.h5$")
_SEQ_RE = re.compile(r"_(\d+)\.h5$")


def _to_bytes_attr(v):
    """h5 attrs 에 문자열을 저장할 때 항상 bytes 로 정규화."""
    if isinstance(v, bytes):
        return v
    if isinstance(v, str):
        return v.encode()
    return v


# ─────────────────────────────────────────────────────────────────────────────
# 0. 베이스 h5 컨테이너
# ─────────────────────────────────────────────────────────────────────────────
class H5DataContainer:
    def __init__(self, datasets=None, attrs=None, default_filename=None):
        self.datasets: dict[str, np.ndarray] = {}
        if datasets:
            for k, v in datasets.items():
                self.datasets[k] = np.asarray(v)
        self.attrs: dict = dict(attrs) if attrs else {}
        self.default_filename = (
            str(default_filename)
            if default_filename is not None
            else Path(__file__).resolve().parent + "/data.h5"
        )

    # ─── dict 비슷한 접근자 ────────────────────────────────────────────────
    def __contains__(self, name):
        return name in self.datasets

    def __getitem__(self, name):
        return self.datasets[name]

    def __setitem__(self, name, value):
        self.datasets[name] = np.asarray(value)

    def keys(self):
        return self.datasets.keys()

    def set_attr(self, key, value):
        self.attrs[key] = value

    def get_attr(self, key, default=None):
        return self.attrs.get(key, default)

    # ─── 저장 / 로드 ────────────────────────────────────────────────────────
    def save(self, out_path=None):
        if out_path is None:
            out_path = self.default_filename
        if out_path is None:
            raise ValueError("저장할 경로 또는 default_filename 이 필요합니다.")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as f:
            for k, v in self.attrs.items():
                f.attrs[k] = v
            for name, arr in self.datasets.items():
                arr = np.asarray(arr)
                if arr.ndim >= 1 and arr.size > 0:
                    f.create_dataset(name, data=arr, compression="gzip")
                else:
                    f.create_dataset(name, data=arr)
        return out_path

    @classmethod
    def load(cls, in_path):
        in_path = Path(in_path)
        attrs = {}
        datasets = {}
        with h5py.File(in_path, "r") as f:
            for k in f.attrs:
                attrs[k] = f.attrs[k]
            for name in f.keys():
                obj = f[name]
                if isinstance(obj, h5py.Dataset):
                    datasets[name] = obj[:]
        return cls(datasets=datasets, attrs=attrs, default_filename=in_path)


# ─────────────────────────────────────────────────────────────────────────────
# 1, 2. Raw / 시간영역 신호 컨테이너
# ─────────────────────────────────────────────────────────────────────────────
class RawData(H5DataContainer):
    CHANNEL_NAMES = CHANNEL_NAMES

    # raw chunk 마다 패킷 수 만큼 1차원으로 들어 있어 chunk 들 사이를 그대로
    # 이어 붙이면 되는 필드들.
    _PER_PACKET_FIELDS = ("fast_flags", "fast_motor_state", "ts_us")
    # chunk 별 길이가 다를 수 있지만 마찬가지로 axis=0 으로 이어 붙이는 필드들.
    _VARLEN_FIELDS = ("events", "slow_ctx")
    # chunk 마다 한 줄짜리. 모든 chunk 분을 그대로 axis=0 으로 쌓아 둔다.
    _SINGLETON_FIELDS = ("lifetime", "motor_spec")
    META_FIELDS = _PER_PACKET_FIELDS + _VARLEN_FIELDS + _SINGLETON_FIELDS

    def __init__(
        self,
        data=None,
        fs_hz=FS_HZ,
        source=None,
        channels=None,
        datasets=None,
        attrs=None,
        default_filename=None,
    ):
        ds = {} if datasets is None else dict(datasets)
        at = {} if attrs is None else dict(attrs)
        if data is not None:
            ds["data"] = np.asarray(data)
        at.setdefault("fs_hz", int(fs_hz))
        names = channels if channels is not None else self.CHANNEL_NAMES
        at.setdefault("channels", np.array(list(names), dtype="S"))
        if source is not None:
            at["source"] = _to_bytes_attr(source)
        super().__init__(datasets=ds, attrs=at, default_filename=default_filename)

    # ─── 데이터 접근 ───────────────────────────────────────────────────────
    @property
    def data(self) -> np.ndarray:
        return self.datasets["data"]

    @property
    def fs_hz(self) -> int:
        return int(self.attrs["fs_hz"])

    @property
    def channels(self) -> list[str]:
        raw = self.attrs.get("channels")
        if raw is None:
            return list(self.CHANNEL_NAMES)
        return [c.decode() if isinstance(c, bytes) else c for c in list(raw)]

    @property
    def source(self) -> str:
        v = self.attrs.get("source", b"raw")
        return v.decode() if isinstance(v, bytes) else str(v)

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    def channel(self, name) -> np.ndarray:
        """채널 이름으로 (N,) 1D 신호 접근."""
        return self.data[self.channels.index(name)]

    # ─── raw chunk 모음 → RawData ─────────────────────────────────────────
    @classmethod
    def from_chunks(cls, files: Iterable[Path]):
        """raw chunk h5 파일들을 시퀀스 순으로 합쳐 RawData 를 만든다.

        ``fast_adc`` 외에도 raw chunk 안의 부수 메타데이터 (events, fast_flags,
        fast_motor_state, lifetime, motor_spec, slow_ctx, ts_us) 와 top-level
        attrs 를 함께 읽어 모두 보존한다.
        """
        chunks = []
        # 메타데이터 필드들은 chunk 별로 모아 두었다가 마지막에 합친다.
        collected: dict[str, list[np.ndarray]] = {f: [] for f in cls.META_FIELDS}
        top_attrs: dict = {}

        for fp in files:
            with h5py.File(fp, "r") as f:
                # 첫 chunk 의 top-level attrs 만 기준으로 잡는다 (chunk 간 동일 가정).
                if not top_attrs:
                    top_attrs = {k: f.attrs[k] for k in f.attrs}
                d = f["fast_adc"][:]
                chunks.append(d.astype(np.uint16).transpose(1, 0, 2).reshape(6, -1))
                for field in cls.META_FIELDS:
                    if field in f and isinstance(f[field], h5py.Dataset):
                        collected[field].append(f[field][:])

        raw = np.concatenate(chunks, axis=1)
        datasets: dict[str, np.ndarray] = {"data": raw}
        for field, arrs in collected.items():
            if not arrs:
                continue
            try:
                datasets[field] = np.concatenate(arrs, axis=0)
            except (ValueError, TypeError):
                # 구조체 dtype 가 chunk 마다 미묘하게 달라 concat 이 실패하면
                # 첫 chunk 값만 보관 (싱글톤 취급).
                datasets[field] = arrs[0]

        # source 만 따로 셋업해 두고, 나머지 attrs 는 raw chunk 에서 가져온
        # 것을 그대로 사용한다.
        top_attrs.setdefault("fs_hz", FS_HZ)
        top_attrs["source"] = _to_bytes_attr("raw")
        return cls(data=None, datasets=datasets, attrs=top_attrs)

    @staticmethod
    def file_seq(path: Path) -> int:
        m = _SEQ_RE.search(path.name)
        return int(m.group(1)) if m else 0

    @staticmethod
    def output_name(path: Path) -> str:
        m = _NAME_RE.match(path.name)
        return m.group(1).split("T")[0] if m else path.name.split("T")[0]


# ─────────────────────────────────────────────────────────────────────────────
# 3. 전처리 함수 (RawData → RawData)
# ─────────────────────────────────────────────────────────────────────────────
#   * "다른 전처리 함수도 계속 추가할 수 있도록 한다" 요구를 만족시키기 위해
#     모든 전처리는 (raw_data: RawData) -> RawData 시그니처로 둔다.
#   * 결과는 새 RawData 를 반환해야 한다 (입력은 immutably 취급).
# ─────────────────────────────────────────────────────────────────────────────
def log16p(raw_data: RawData) -> RawData:
    scaled = np.log2(raw_data.data.astype(np.float64) + 1) / 16.0
    new_datasets = {k: v for k, v in raw_data.datasets.items() if k != "data"}
    new_datasets["data"] = scaled
    new_attrs = dict(raw_data.attrs)
    new_attrs["source"] = _to_bytes_attr("scaled_log16p")
    return RawData(data=None, datasets=new_datasets, attrs=new_attrs)


# 향후 전처리 함수 추가 예 (시그니처를 동일하게 유지):
#   def zscore(raw_data: RawData) -> RawData: ...
#   def lowpass(raw_data: RawData, cutoff_hz: float) -> RawData: ...


# ─────────────────────────────────────────────────────────────────────────────
# 4, 5. FFT 결과 컨테이너 + 정적 변환 메서드
# ─────────────────────────────────────────────────────────────────────────────
class FFTData(H5DataContainer):
    """채널별 단측 진폭 스펙트럼 컨테이너.

    저장 형식
    --------
    /freqs   : (n_bins,) 주파수 축 [Hz]
    /data    : (n_ch, n_win, n_bins) 단측 진폭 스펙트럼 — 6채널을 한 데이터셋에
               담고, 첫 축의 0~5 인덱스가 각 채널 (attrs['channels'] 순서).
    attrs:
      fs_hz                : 샘플링 주파수
      samples_per_window   : FFT 청크 길이 (샘플)
      window               : 창함수 이름 (현재 "hann")
      normalization        : "amplitude_single_sided"
      channels             : 채널 이름 목록 — 축 0 인덱스 매핑
      source               : 어느 RawData 에서 만든 것인지 표시
    """

    CHANNEL_NAMES = CHANNEL_NAMES

    def __init__(
        self,
        fft_mag=None,
        freqs=None,
        fs_hz=FS_HZ,
        samples_per_window=None,
        window=b"hann",
        normalization=b"amplitude_single_sided",
        source=b"scaled",
        channels=None,
        datasets=None,
        attrs=None,
        default_filename=None,
    ):
        ds = {} if datasets is None else dict(datasets)
        at = {} if attrs is None else dict(attrs)
        names = list(channels) if channels is not None else list(self.CHANNEL_NAMES)
        if fft_mag is not None:
            # 6채널 스펙트럼을 채널별 데이터셋으로 쪼개지 않고 한 덩어리로 저장.
            ds["data"] = np.asarray(fft_mag)
        if freqs is not None:
            ds["freqs"] = np.asarray(freqs)
        at.setdefault("fs_hz", int(fs_hz))
        if samples_per_window is not None:
            at.setdefault("samples_per_window", int(samples_per_window))
        at.setdefault("window", _to_bytes_attr(window))
        at.setdefault("normalization", _to_bytes_attr(normalization))
        at.setdefault("source", _to_bytes_attr(source))
        at.setdefault("channels", np.array(names, dtype="S"))
        super().__init__(datasets=ds, attrs=at, default_filename=default_filename)

    # ─── 데이터 접근 ───────────────────────────────────────────────────────
    @property
    def channels(self) -> list[str]:
        raw = self.attrs.get("channels")
        if raw is None:
            return list(self.CHANNEL_NAMES)
        return [c.decode() if isinstance(c, bytes) else c for c in list(raw)]

    @property
    def freqs(self) -> np.ndarray:
        return self.datasets["freqs"]

    @property
    def samples_per_window(self) -> int:
        return int(self.attrs["samples_per_window"])

    @property
    def fs_hz(self) -> int:
        return int(self.attrs["fs_hz"])

    @property
    def fft_mag(self) -> np.ndarray:
        """(n_ch, n_win, n_bins) 스펙트럼 덩어리."""
        return self.datasets["data"]

    def channel(self, name) -> np.ndarray:
        """채널 이름으로 (n_win, n_bins) 스펙트럼 접근."""
        return self.datasets["data"][self.channels.index(name)]

    # ─── Raw → FFT 정적 변환 (실행사항 5) ──────────────────────────────────
    @staticmethod
    def from_raw(
        raw_data: RawData, samples_per_window: int = FFT_WINDOW_SAMPLES
    ) -> "FFTData":
        """RawData 의 시간영역 신호를 윈도우 단위로 FFT 해 FFTData 를 생성한다.

        * Hann 창으로 스펙트럼 누수를 줄이고
        * "동일 진폭의 정현파라면 막대 높이가 그 진폭에 가깝다" 가 성립하도록
          ``|X[k]| * 2 / Σ window`` 단측 정규화를 적용한다 (DC·Nyquist 는 *2 제외).

        이 정규화 덕분에 60 Hz / 120 Hz / 180 Hz 등 고조파를 막대 그래프
        높이만 보고 직관적으로 비교/검출할 수 있다.
        """
        data = raw_data.data
        fs = raw_data.fs_hz
        n_total = data.shape[1]
        n_win = n_total // samples_per_window
        spw = samples_per_window
        if n_win == 0:
            # 신호가 한 청크보다 짧다면 통째로 한 윈도우로 처리.
            n_win = 1
            spw = n_total

        n_bins = spw // 2 + 1
        freqs = np.fft.rfftfreq(spw, d=1.0 / fs)
        win_fn = np.hanning(spw)
        coherent_gain = win_fn.sum()

        n_chan = data.shape[0]
        out = np.empty((n_chan, n_win, n_bins), dtype=np.float64)
        for w in range(n_win):
            sl = slice(w * spw, (w + 1) * spw)
            chunk = data[:, sl].astype(np.float64)
            chunk = chunk - chunk.mean(axis=1, keepdims=True)
            chunk = chunk * win_fn
            X = np.fft.rfft(chunk, axis=1)
            mag = np.abs(X) * (2.0 / coherent_gain)
            mag[:, 0] *= 0.5
            if spw % 2 == 0:
                mag[:, -1] *= 0.5
            out[:, w, :] = mag

        # RawData 의 부수 데이터 (events, fast_flags, lifetime, motor_spec,
        # slow_ctx, ts_us) 를 그대로 통과시켜 둔다. 알맞은 정렬은 호출자가
        # 별도로 판단해야 하지만, 메타데이터 자체는 잃지 않는다.
        extra_datasets = {k: v for k, v in raw_data.datasets.items() if k != "data"}
        extra_attrs = {
            k: v for k, v in raw_data.attrs.items() if k not in {"source", "channels"}
        }
        return FFTData(
            fft_mag=out,
            freqs=freqs,
            fs_hz=fs,
            samples_per_window=spw,
            source=raw_data.source,
            channels=raw_data.channels,
            datasets=extra_datasets,
            attrs=extra_attrs,
        )

    # ─── 역변환 (디버깅/시각화 보조) ───────────────────────────────────────
    @staticmethod
    def inverse_fft(fft_data: "FFTData") -> np.ndarray:
        """진폭 스펙트럼만으로 시간영역 신호를 재구성 (위상 0 가정).

        본 클래스가 저장하는 데이터는 위상 정보가 빠진 진폭 스펙트럼이므로
        완전한 역변환은 불가능하다. 이 함수는 위상을 0 으로 가정해 형태만
        복원하는 근사 IFFT 이며, 검사/시각화 용도로만 의미가 있다.
        반환: (n_ch, n_win, samples_per_window) ndarray.
        """
        spw = fft_data.samples_per_window
        fft_mag = fft_data.fft_mag  # (n_ch, n_win, n_bins)
        n_chan, n_win, _ = fft_mag.shape

        out = np.empty((n_chan, n_win, spw), dtype=np.float64)
        for i in range(n_chan):
            mag = fft_mag[i].astype(np.float64).copy()
            # from_raw 의 *2 보정을 되돌려 |X[k]| 로 환원.
            mag[:, 0] *= 2.0
            if spw % 2 == 0:
                mag[:, -1] *= 2.0
            X = (
                mag / 2.0
            )  # *2 / Σwin 의 역 — 단, Σwin 은 IFFT 후 보정 불필요(상대형상만 필요).
            for w in range(n_win):
                out[i, w] = np.fft.irfft(X[w], n=spw)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Cross-correlation 공통 도움 함수
# ─────────────────────────────────────────────────────────────────────────────
def _normalized_xcorr_pair(xa: np.ndarray, xb: np.ndarray, max_lag: int) -> np.ndarray:
    """동일 길이의 두 1D 신호로부터 ±max_lag 범위의 정규 cross-corr 추출.

    ┌─ Cross-correlation 짧은 강의 ─────────────────────────────────────────┐
    │ 두 신호 x[n], y[n] 가 있을 때 R_xy[k] = Σ_n x[n+k] · y[n] 가 정의.  │
    │ k > 0 에서 피크 → x 가 y 보다 k 샘플 지연(lag) 되어 있다는 뜻.       │
    │ k < 0 에서 피크 → x 가 y 를 |k| 만큼 앞선다.                         │
    │                                                                       │
    │ 절대 크기를 정규화하기 위해 윈도우마다 평균을 빼고 분산의 기하평균   │
    │ 으로 나눠 Pearson 상관계수 형태로 만든다 (값은 [-1, +1]).            │
    └────────────────────────────────────────────────────────────────────────┘
    """
    n = xa.shape[0]
    center = n - 1
    xa = xa.astype(np.float64, copy=True)
    xb = xb.astype(np.float64, copy=True)
    xa -= xa.mean()
    xb -= xb.mean()
    denom = np.sqrt((xa * xa).sum() * (xb * xb).sum())
    if denom < 1e-12:
        return np.zeros(2 * max_lag + 1, dtype=np.float64)
    c = correlate(xa, xb, mode="full", method="fft") / denom
    return c[center - max_lag : center + max_lag + 1]


# ─────────────────────────────────────────────────────────────────────────────
# 6, 8. 시간영역 cross-correlation 결과 컨테이너
# ─────────────────────────────────────────────────────────────────────────────
class XcorrTimeData(H5DataContainer):
    """시간영역 9쌍 cross-correlation 결과 컨테이너.

    저장 형식
    --------
    /lags   : (2L+1,) 샘플 단위 lag 축
    /xcorr  : (n_pair, n_win, 2L+1) 정규 cross-correlation — 9쌍을 한 데이터셋에
              담고, 첫 축의 0~8 인덱스가 각 쌍 (attrs['pairs'] 순서).
    attrs:
      samples_per_window : 윈도우 크기 [샘플]
      max_lag            : 저장한 lag 의 최대 절댓값 [샘플]
      pairs              : 채널 쌍 이름 목록 (예: b'v1-v2', …) — 축 0 인덱스 매핑
      source             : 어느 신호에서 만든 xcorr 인지 (예: "scaled_log16p")
      lag_unit           : "sample"
    """

    XCORR_PAIRS = XCORR_PAIRS

    def __init__(
        self,
        xcorr_dict=None,
        lags=None,
        samples_per_window=None,
        max_lag=None,
        source=b"scaled",
        fs_hz=FS_HZ,
        datasets=None,
        attrs=None,
        default_filename=None,
    ):
        ds = {} if datasets is None else dict(datasets)
        at = {} if attrs is None else dict(attrs)
        if xcorr_dict is not None:
            # 9쌍을 쌍별 데이터셋으로 쪼개지 않고 한 덩어리 (n_pair, …) 로 저장.
            pair_names = list(xcorr_dict.keys())
            ds["xcorr"] = np.stack(
                [np.asarray(xcorr_dict[n]) for n in pair_names], axis=0
            )
            at.setdefault("pairs", np.array(pair_names, dtype="S"))
        if lags is not None:
            ds["lags"] = np.asarray(lags)
        if samples_per_window is not None:
            at.setdefault("samples_per_window", int(samples_per_window))
        if max_lag is not None:
            at.setdefault("max_lag", int(max_lag))
        at.setdefault("fs_hz", int(fs_hz))
        at.setdefault("source", _to_bytes_attr(source))
        at.setdefault("lag_unit", b"sample")
        super().__init__(datasets=ds, attrs=at, default_filename=default_filename)

    # ─── 데이터 접근 ───────────────────────────────────────────────────────
    @property
    def lags(self) -> np.ndarray:
        return self.datasets["lags"]

    @property
    def xcorr(self) -> np.ndarray:
        """(n_pair, n_win, 2L+1) cross-correlation 덩어리."""
        return self.datasets["xcorr"]

    @property
    def pairs(self) -> list[str]:
        raw = self.attrs.get("pairs")
        if raw is not None:
            return [p.decode() if isinstance(p, bytes) else p for p in list(raw)]
        return [f"{a}-{b}" for a, b in self.XCORR_PAIRS]

    def pair_at(self, idx: int) -> np.ndarray:
        """인덱스 (0..8) 로 (n_win, 2L+1) 행렬 접근."""
        return self.datasets["xcorr"][idx]

    def pair(self, name: str) -> np.ndarray:
        """채널 쌍 이름 (예: 'v1-v2') 으로 (n_win, 2L+1) 행렬 접근."""
        return self.pair_at(self.pairs.index(name))

    def lag_seconds(self) -> np.ndarray:
        """lag 축을 초 단위로 변환."""
        return self.lags / float(self.attrs["fs_hz"])

    # ─── RawData → XcorrTimeData 정적 변환 (실행사항 8) ────────────────────
    @staticmethod
    def from_signal(
        raw_data: RawData,
        samples_per_window: int = SAMPLES_PER_WINDOW,
        max_lag: int = MAX_LAG_SAMPLES,
        pairs: Iterable[tuple[str, str]] = XCORR_PAIRS,
    ) -> "XcorrTimeData":
        """RawData (raw 또는 log16p scaled 모두 가능) 의 9쌍 시간영역 xcorr.

        데이터 길이가 매우 길 때 한 번에 xcorr 를 구하면 시간 변화 정보가
        사라지므로 ``samples_per_window`` 단위로 잘라 윈도우마다 따로 계산하고,
        결과 행렬을 ``(n_win, 2·max_lag + 1)`` 로 쌓는다. 잘라지지 않는 자투리
        샘플은 버린다. 입력의 부수 메타데이터는 결과로 그대로 보존한다.
        """
        data = raw_data.data
        channels = raw_data.channels
        name_to_idx = {n: i for i, n in enumerate(channels)}

        n_total = data.shape[1]
        n_win = n_total // samples_per_window
        lags = np.arange(-max_lag, max_lag + 1)

        result: dict[str, np.ndarray] = {}
        for a, b in pairs:
            ia, ib = name_to_idx[a], name_to_idx[b]
            out = np.empty((n_win, lags.size), dtype=np.float64)
            for w in range(n_win):
                sl = slice(w * samples_per_window, (w + 1) * samples_per_window)
                out[w] = _normalized_xcorr_pair(
                    data[ia, sl],
                    data[ib, sl],
                    max_lag,
                )
            result[f"{a}-{b}"] = out

        extra_datasets = {k: v for k, v in raw_data.datasets.items() if k != "data"}
        extra_attrs = {
            k: v for k, v in raw_data.attrs.items() if k not in {"source", "channels"}
        }
        return XcorrTimeData(
            xcorr_dict=result,
            lags=lags,
            samples_per_window=samples_per_window,
            max_lag=max_lag,
            fs_hz=raw_data.fs_hz,
            source=raw_data.source,
            datasets=extra_datasets,
            attrs=extra_attrs,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7, 8. FFT cross-correlation 결과 컨테이너
# ─────────────────────────────────────────────────────────────────────────────
class XcorrFFTData(H5DataContainer):
    """FFT 진폭 스펙트럼 9쌍 cross-correlation 결과 컨테이너.

    저장 형식 (시간영역 xcorr 와 유사하지만 lag 단위가 bin):
      /lags   : (2L+1,) bin 단위 lag 축
      /xcorr  : (n_pair, n_win, 2L+1) 정규 cross-correlation — 9쌍을 한 데이터셋
                에 담고, 첫 축의 0~8 인덱스가 각 쌍 (attrs['pairs'] 순서).
      attrs:
        samples_per_window : FFT 청크 길이 (lag → Hz 환산용)
        max_lag            : ±max_lag bin
        fs_hz              : 샘플링 주파수
        pairs              : 채널 쌍 이름 목록 — 축 0 인덱스 매핑
        source, lag_unit("bin")
    """

    XCORR_PAIRS = XCORR_PAIRS

    def __init__(
        self,
        xcorr_dict=None,
        lags=None,
        samples_per_window=None,
        max_lag=None,
        fs_hz=FS_HZ,
        source=b"scaled",
        datasets=None,
        attrs=None,
        default_filename=None,
    ):
        ds = {} if datasets is None else dict(datasets)
        at = {} if attrs is None else dict(attrs)
        if xcorr_dict is not None:
            pair_names = list(xcorr_dict.keys())
            ds["xcorr"] = np.stack(
                [np.asarray(xcorr_dict[n]) for n in pair_names], axis=0
            )
            at.setdefault("pairs", np.array(pair_names, dtype="S"))
        if lags is not None:
            ds["lags"] = np.asarray(lags)
        if samples_per_window is not None:
            at.setdefault("samples_per_window", int(samples_per_window))
        if max_lag is not None:
            at.setdefault("max_lag", int(max_lag))
        at.setdefault("fs_hz", int(fs_hz))
        at.setdefault("source", _to_bytes_attr(source))
        at.setdefault("lag_unit", b"bin")
        super().__init__(datasets=ds, attrs=at, default_filename=default_filename)

    @property
    def lags(self) -> np.ndarray:
        return self.datasets["lags"]

    @property
    def xcorr(self) -> np.ndarray:
        """(n_pair, n_win, 2L+1) cross-correlation 덩어리."""
        return self.datasets["xcorr"]

    @property
    def pairs(self) -> list[str]:
        raw = self.attrs.get("pairs")
        if raw is not None:
            return [p.decode() if isinstance(p, bytes) else p for p in list(raw)]
        return [f"{a}-{b}" for a, b in self.XCORR_PAIRS]

    def pair_at(self, idx: int) -> np.ndarray:
        """인덱스 (0..8) 로 (n_win, 2L+1) 행렬 접근."""
        return self.datasets["xcorr"][idx]

    def pair(self, name: str) -> np.ndarray:
        return self.pair_at(self.pairs.index(name))

    def lag_hz(self) -> np.ndarray:
        """bin 단위 lag 축을 Hz 단위로 변환."""
        fs = float(self.attrs["fs_hz"])
        spw = float(self.attrs["samples_per_window"])
        return self.lags * (fs / spw)

    # ─── FFTData → XcorrFFTData 정적 변환 (실행사항 8) ─────────────────────
    @staticmethod
    def from_fft(
        fft_data: FFTData,
        max_lag: int = FFT_MAX_LAG_BINS,
        pairs: Iterable[tuple[str, str]] = XCORR_PAIRS,
    ) -> "XcorrFFTData":
        """FFTData 의 9쌍 cross-correlation.

        피크 lag = "두 채널의 평균적인 주파수 차이 (bin)". 정상적인 3상 신호는
        v-v / i-i 가 lag=0 에 강한 피크를 보여야 한다. 입력 FFTData 가 들고
        있는 부수 메타데이터는 그대로 통과시킨다.
        """
        fft_mag = fft_data.fft_mag  # (n_ch, n_win, n_bins)
        channels = fft_data.channels
        name_to_idx = {n: i for i, n in enumerate(channels)}
        _, n_win, _ = fft_mag.shape
        lags = np.arange(-max_lag, max_lag + 1)

        result: dict[str, np.ndarray] = {}
        for a, b in pairs:
            ia, ib = name_to_idx[a], name_to_idx[b]
            out = np.empty((n_win, lags.size), dtype=np.float64)
            for w in range(n_win):
                out[w] = _normalized_xcorr_pair(
                    fft_mag[ia, w],
                    fft_mag[ib, w],
                    max_lag,
                )
            result[f"{a}-{b}"] = out

        # fft_data 의 부수 데이터셋을 통과 — 단 FFTData 의 스펙트럼/주파수
        # 데이터셋은 XcorrFFTData 와 의미가 다르므로 제외한다.
        extra_datasets = {
            k: v for k, v in fft_data.datasets.items() if k not in {"data", "freqs"}
        }
        exclude_attrs = {
            "source",
            "channels",
            "window",
            "normalization",
            "samples_per_window",
        }
        extra_attrs = {
            k: v for k, v in fft_data.attrs.items() if k not in exclude_attrs
        }
        return XcorrFFTData(
            xcorr_dict=result,
            lags=lags,
            samples_per_window=fft_data.samples_per_window,
            max_lag=max_lag,
            fs_hz=fft_data.fs_hz,
            source=fft_data.attrs.get("source", b"scaled"),
            datasets=extra_datasets,
            attrs=extra_attrs,
        )


__all__ = [
    "CHANNEL_NAMES",
    "XCORR_PAIRS",
    "FS_HZ",
    "SAMPLES_PER_WINDOW",
    "MAX_LAG_SAMPLES",
    "FFT_WINDOW_SAMPLES",
    "FFT_MAX_LAG_BINS",
    "H5DataContainer",
    "RawData",
    "log16p",
    "FFTData",
    "XcorrTimeData",
    "XcorrFFTData",
]
