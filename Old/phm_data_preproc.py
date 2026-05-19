from pathlib import Path
import re
import sys

import h5py
import numpy as np
from scipy.signal import correlate

CHANNEL_NAMES = ["v1", "v2", "v3", "i1", "i2", "i3"]
FS_HZ = 20000          # ADC 샘플링 [Hz]
WINDOW_MS = 50         # 시간영역 cross-correlation 청크 길이 [ms]
SAMPLES_PER_WINDOW = FS_HZ * WINDOW_MS // 5000  # 1000

# ─────────────────────────────────────────────────────────────────────────────
# FFT 설정 (실행사항 5)
# ─────────────────────────────────────────────────────────────────────────────
# 실행사항 5 의 요구: "샘플링 데이터는 100,000개를 기준으로" FFT 를 한다.
# fs = 20kHz 이므로 한 윈도우 길이는 100,000 / 20,000 = 5 초.
# 0 ~ fs/2 = 10kHz 범위를 50,001 개 bin 으로 표현 → 1 bin = fs/N = 0.2 Hz.
# 60Hz 기본파와 그 고조파(120, 180, 240, …)를 모두 충분히 분리해 볼 수 있다.
FFT_WINDOW_SAMPLES = 100_000     # FFT 한 청크의 샘플 수
# FFT 결과끼리 cross-correlation 할 때 저장할 lag (= 주파수 이동) 의 최대값.
# bin 단위이며, ±500 bin = ±100 Hz 정도면 위상/주파수 매칭 분석에 충분하다.
FFT_MAX_LAG_BINS = 500

# ─────────────────────────────────────────────────────────────────────────────
# Cross-correlation 설정 (실행사항 3)
# ─────────────────────────────────────────────────────────────────────────────
# 9개의 채널 쌍에 대해 윈도우 단위로 cross-correlation 을 계산한다.
# - v-v 쌍 (3상 전압 사이 위상 관계)
# - i-i 쌍 (3상 전류 사이 위상 관계)
# - v-i 쌍 (전압-전류 사이 위상 관계, 곧 역률 관련)
XCORR_PAIRS = [
    ("v1", "v2"), ("v2", "v3"), ("v3", "v1"),
    ("i1", "i2"), ("i2", "i3"), ("i3", "i1"),
    ("v1", "i1"), ("v2", "i2"), ("v3", "i3"),
]
# 결과로 저장할 lag (시간차) 의 최대값. 단위는 샘플.
#   SAMPLES_PER_WINDOW = 1000, fs = 20kHz ⇒ ±500 샘플 = ±25 ms
# 60Hz 신호의 주기는 약 16.67 ms (≈ 333 샘플) 이므로 ±25 ms 범위면
# 한 주기 이상을 보여줄 수 있어, 위상차/주기 구조를 모두 관찰할 수 있다.
MAX_LAG_SAMPLES = SAMPLES_PER_WINDOW // 2

_NAME_RE = re.compile(r"^([^_]+_[^_]+_[^_]+)_.+\.h5$")
_SEQ_RE = re.compile(r"_(\d+)\.h5$")

def file_seq(path):
  m = _SEQ_RE.search(path.name)
  return int(m.group(1)) if m else 0

def output_name(path):
  m = _NAME_RE.match(path.name)
  return m.group(1).split('T')[0] if m else path.name.split('T')[0]

def load_phases(path):
  """h5 파일에서 (6, N) 채널별 파형을 추출.
  fast_adc shape (n_packets, 6, 50) → transpose(1,0,2).reshape(6, -1).
  """
  with h5py.File(path, "r") as f:
    d = f["fast_adc"][:]
  return d.astype(np.uint16).transpose(1, 0, 2).reshape(6, -1)

def log16p(x):
  return np.log2(np.asarray(x, dtype=np.float64) + 1) / 16.0


def cross_correlate_pairs(scaled, samples_per_window, max_lag):
  """9개 채널 쌍에 대한 윈도우 단위 정규 cross-correlation 을 계산한다.

  ┌─ Cross-correlation 짧은 강의 ──────────────────────────────────────────┐
  │ 두 신호 x[n], y[n] 가 있을 때, "x 를 시간축으로 k 샘플만큼 옮긴 뒤    │
  │  y 와 곱해서 모두 더한 값" 을 cross-correlation 이라고 한다.          │
  │                                                                       │
  │       R_xy[k] = Σ_n  x[n + k] · y[n]                                  │
  │                                                                       │
  │  k 를 "lag" 라고 부르며 단위는 샘플이다. k 를 -K..+K 까지 바꿔가며    │
  │  R_xy[k] 의 값을 모두 계산하면, "두 신호가 시간차 얼마에서 가장 잘    │
  │  맞아 떨어지는가" 를 보여주는 곡선이 나온다.                          │
  │                                                                       │
  │  ▸ R_xy[0] : 두 신호를 그대로 곱해서 더한 값 (정렬되어 있을 때 닮음).│
  │  ▸ k = L > 0 에서 피크 : x 를 미래쪽으로 L 칸 밀어야 y 와 맞다는 뜻.  │
  │      즉, x 가 y 보다 L 샘플 지연(lag) 되어 있다.                      │
  │  ▸ k = L < 0 에서 피크 : 반대로 x 가 y 를 |L| 샘플 앞선다.            │
  │                                                                       │
  │ 정규화 (normalization)                                                │
  │  원본 신호의 절대 크기에 따라 R 값이 휙휙 바뀌므로, 실제 분석에서는   │
  │  보통 윈도우마다 평균을 빼고 분산으로 나눠 [-1, +1] 범위의 Pearson    │
  │  상관계수 형태로 바꿔 쓴다. 본 함수도 이 방식을 사용한다.             │
  │                                                                       │
  │      ρ_xy[k] = Σ (x[n+k] - x̄)(y[n] - ȳ)                              │
  │                ────────────────────────────────                       │
  │                √( Σ(x[n]-x̄)²  ·  Σ(y[n]-ȳ)² )                         │
  │                                                                       │
  │ 윈도우(window) 단위 처리                                              │
  │  데이터 길이가 매우 길고, 모터 상태도 시간에 따라 바뀌므로 한 번에    │
  │  전체에 대해 cross-correlation 을 구하면 시간 변화 정보가 사라진다.   │
  │  따라서 samples_per_window 만큼 잘라(잘려나간 자투리는 버린다) 각     │
  │  윈도우마다 따로 cross-correlation 을 계산한다. 결과는                │
  │  (윈도우 수, 2·max_lag + 1) 모양의 2D 배열이 되고, 이를 그대로        │
  │  히트맵으로 그리면 "시간 × lag" 의 상관관계 지도가 된다.              │
  └────────────────────────────────────────────────────────────────────────┘

  구현 메모
  --------
  * scipy.signal.correlate(a, b, mode='full', method='fft')
    - full mode 결과 길이는 (N + N - 1) = 2N - 1.
    - 중앙 인덱스 (samples_per_window - 1) 가 lag = 0 에 해당.
    - method='fft' 는 푸리에 변환을 이용해 O(N log N) 으로 계산.
      윈도우 N = 1000 정도에서도 직접 합산 대비 충분히 빠르다.
  * 두 신호의 길이는 윈도우 내에서 항상 같으므로(둘 다 N 샘플),
    lag 의 정의가 명확히 위에 적은 식과 일치한다.
  * 분모가 0 인 경우(완전 평탄 신호) 는 ρ 정의 자체가 어긋나므로 0 으로 채운다.

  Parameters
  ----------
  scaled : ndarray, shape (6, N)
    log16p 로 정규화된 채널별 신호 (행 순서는 CHANNEL_NAMES 와 동일).
  samples_per_window : int
    한 윈도우의 샘플 수.
  max_lag : int
    결과로 저장할 lag 의 최대 절댓값(샘플). 결과 두번째 축 길이는
    2·max_lag + 1 이 된다.

  Returns
  -------
  result : dict[str, ndarray]
    채널 쌍 이름 ("v1-v2" 등) 을 키로, (n_win, 2·max_lag + 1) 모양의
    정규 cross-correlation 행렬을 값으로 하는 딕셔너리.
  lags : ndarray, shape (2·max_lag + 1,)
    lag 축의 값을 샘플 단위로 담은 배열. fs 로 나누면 초 단위가 된다.
  """
  # 채널 이름을 행 인덱스로 변환하기 위한 매핑
  name_to_idx = {n: i for i, n in enumerate(CHANNEL_NAMES)}

  N_total = scaled.shape[1]                       # 전체 샘플 수
  n_win = N_total // samples_per_window           # 잘라지는 윈도우 개수 (자투리는 버림)
  lags = np.arange(-max_lag, max_lag + 1)         # 결과 lag 축 (샘플 단위)
  # full cross-corr 결과의 중앙 인덱스. 이 위치가 lag = 0 에 해당한다.
  center = samples_per_window - 1

  result = {}
  for a_name, b_name in XCORR_PAIRS:
    ia = name_to_idx[a_name]
    ib = name_to_idx[b_name]
    # 각 쌍별 출력 행렬: (윈도우 수, lag 개수)
    out = np.empty((n_win, lags.size), dtype=np.float64)

    for w in range(n_win):
      sl = slice(w * samples_per_window, (w + 1) * samples_per_window)
      xa = scaled[ia, sl].astype(np.float64)
      xb = scaled[ib, sl].astype(np.float64)

      # 1) 윈도우 내 평균 제거 (DC 성분 제거). 평균이 큰 신호는
      #    상관관계와 무관한 곱셈 항이 결과를 지배할 수 있다.
      xa -= xa.mean()
      xb -= xb.mean()

      # 2) 정규화 분모: 두 신호 분산의 기하평균. 이 값으로 나누면
      #    결과 ρ 가 [-1, +1] 범위에 들어가게 된다.
      denom = np.sqrt((xa * xa).sum() * (xb * xb).sum())
      if denom < 1e-12:
        # 신호가 완전히 평탄해서 분산이 거의 0 인 경우 처리
        out[w] = 0.0
        continue

      # 3) full cross-correlation 계산. 결과 길이 = 2N - 1.
      #    scipy 는 R_xy[k] = Σ_n a[n+k] · b[n] 규약을 따른다.
      c = correlate(xa, xb, mode='full', method='fft') / denom

      # 4) 관심 lag 범위 (-max_lag ~ +max_lag) 만 잘라 저장.
      out[w] = c[center - max_lag : center + max_lag + 1]

    result[f"{a_name}-{b_name}"] = out

  return result, lags


# cross-correlation 결과를 저장하는 그룹 이름. raw / scaled 각각에 대해
# 따로 계산해 두 그룹에 분리 저장한다.
XCORR_GROUP_RAW = "xcorr_raw"
XCORR_GROUP_SCALED = "xcorr_scaled"

# 실행사항 5 결과 그룹
FFT_GROUP_SCALED = "fft_scaled"               # FFT 단측 진폭 스펙트럼
XCORR_FFT_GROUP_SCALED = "xcorr_fft_scaled"   # FFT 결과 쌍 간의 cross-correlation


def _write_xcorr(h5file, group_name, xcorr_dict, lags, samples_per_window, max_lag):
  """이미 열린 h5py.File 핸들에 cross-correlation 결과를 저장한다.

  그룹 구조 (예: group_name = "xcorr_scaled"):
    /xcorr_scaled
      attrs:
        max_lag (int)            : 저장한 lag 의 최대 절댓값(샘플)
        samples_per_window (int) : 윈도우 크기(샘플)
        pairs (S, ...)           : 채널 쌍 이름 목록
        source (S)               : "raw" 또는 "scaled" (어떤 원본
                      데이터에서 만든 cross-corr 인지)
      datasets:
        lags  : (2·max_lag + 1,) 샘플 단위 lag 축
        v1-v2 : (n_win, 2·max_lag + 1) 정규 cross-corr
        v2-v3 : ...
        ...
  """
  # 같은 이름의 그룹이 이미 있으면 새 데이터로 갱신
  if group_name in h5file:
    del h5file[group_name]

  g = h5file.create_group(group_name)
  g.attrs["max_lag"] = max_lag
  g.attrs["samples_per_window"] = samples_per_window
  g.attrs["pairs"] = np.array(list(xcorr_dict.keys()), dtype="S")
  # group_name 에서 "xcorr_" 접두어를 떼어 source 이름으로 기록
  source = group_name[len("xcorr_"):] if group_name.startswith("xcorr_") else group_name
  g.attrs["source"] = np.bytes_(source)
  g.create_dataset("lags", data=lags)
  for name, arr in xcorr_dict.items():
    g.create_dataset(name, data=arr, compression="gzip")


def _compute_and_write_all_xcorr(h5file, raw, scaled, samples_per_window, max_lag):
  """raw 와 scaled 각각에 대해 cross-correlation 을 계산해 두 그룹으로 저장."""
  # raw 는 정수형(uint16) 이지만 cross_correlate_pairs 내부에서 float64 로
  # 변환되므로 그대로 넘겨도 안전하다.
  xc_raw, lags_raw = cross_correlate_pairs(raw, samples_per_window, max_lag)
  _write_xcorr(h5file, XCORR_GROUP_RAW, xc_raw, lags_raw,
               samples_per_window, max_lag)

  xc_scaled, lags_scaled = cross_correlate_pairs(scaled, samples_per_window, max_lag)
  _write_xcorr(h5file, XCORR_GROUP_SCALED, xc_scaled, lags_scaled,
               samples_per_window, max_lag)

  # 이전 버전에서 만든 단일 "/xcorr" 그룹이 남아 있으면 정리.
  if "xcorr" in h5file:
    del h5file["xcorr"]


# ─────────────────────────────────────────────────────────────────────────────
# 실행사항 5 : FFT 변환 + FFT 결과끼리의 cross-correlation
# ─────────────────────────────────────────────────────────────────────────────
def compute_fft_per_channel(scaled, samples_per_window, fs):
  """scaled 데이터를 ``samples_per_window`` 단위 청크로 잘라 채널별 FFT 진폭
  스펙트럼을 만든다.

  개요
  ----
  * 청크당 100,000 샘플 (실행사항 5 요구) → 1 청크 = 5 초 분량.
  * 각 청크에서 채널별로 평균(DC 성분)을 빼고 Hann 창을 곱한 뒤 실수 FFT
    ``np.fft.rfft`` 를 적용한다. 실수 신호이므로 단측(0 ~ fs/2) 스펙트럼만
    유지하면 충분하다.
  * 정규화 (harmonic detection 용이) :

      mag[k]  =  |X[k]| * 2 / Σ window

    ─ ``Σ window`` 는 Hann 창의 coherent gain. 같은 진폭의 정현파라면
    창의 형태와 무관하게 동일한 mag 값을 얻을 수 있도록 보정.
    ─ ``*2`` 는 단측 스펙트럼이 양/음 주파수 양쪽 성분을 합쳐 표시하기 위함.
    (DC 와 Nyquist 빈은 대칭 쌍이 없으므로 *2 를 다시 1/2 로 되돌린다.)
    → 결과 mag[k] 는 입력 신호의 해당 주파수 성분이 가진 "진폭" 의
    근사값이 되어, 60Hz 와 그 고조파 위치에 솟은 막대 높이를 직관적으로
    해석할 수 있다.

  Returns
  -------
  fft_mag : ndarray, shape (6, n_win, n_bins)
    채널 × 윈도우 × 주파수빈 단측 진폭 스펙트럼.
  freqs : ndarray, shape (n_bins,)
    주파수 축 [Hz]. ``np.fft.rfftfreq`` 가 만들어주는 값이다.
  """
  N_total = scaled.shape[1]
  n_win = N_total // samples_per_window
  if n_win == 0:
    # 신호가 너무 짧으면 한 윈도우로 처리. (실 데이터에서는 거의 발생 X)
    n_win = 1
    samples_per_window = N_total

  n_bins = samples_per_window // 2 + 1
  freqs = np.fft.rfftfreq(samples_per_window, d=1.0 / fs)

  n_chan = scaled.shape[0]
  out = np.empty((n_chan, n_win, n_bins), dtype=np.float64)

  # 스펙트럼 누수(leakage) 를 줄이기 위해 Hann 창을 사용한다.
  # 사이드 로브가 낮아 고조파 피크 검출에 유리하다.
  win_fn = np.hanning(samples_per_window)
  coherent_gain = win_fn.sum()

  for w in range(n_win):
    sl = slice(w * samples_per_window, (w + 1) * samples_per_window)
    chunk = scaled[:, sl].astype(np.float64)
    # 채널별 평균 제거 → DC 성분이 다른 주파수 검출을 가리지 않도록.
    chunk = chunk - chunk.mean(axis=1, keepdims=True)
    # 모든 채널에 동일한 창 함수 적용
    chunk = chunk * win_fn
    X = np.fft.rfft(chunk, axis=1)
    mag = np.abs(X) * (2.0 / coherent_gain)
    # DC(0Hz) 와 Nyquist 빈은 단측 변환에서 *2 보정 대상이 아니다.
    mag[:, 0] *= 0.5
    if samples_per_window % 2 == 0:
      mag[:, -1] *= 0.5
    out[:, w, :] = mag

  return out, freqs


def cross_correlate_fft_pairs(fft_mag, max_lag):
  """FFT 진폭 스펙트럼끼리의 윈도우 단위 정규 cross-correlation.

  개념
  ----
  시간영역 cross-correlation 이 "시간 lag 만큼 옮긴 두 파형이 얼마나 닮았나"
  를 보는 것이라면, 주파수영역 cross-correlation 은 "주파수 lag 만큼 옮긴
  두 스펙트럼이 얼마나 닮았나" 를 본다.
    ρ_xy[k] = Pearson( X_a , shift(X_b, k) ),     k 는 bin 단위.

  의미
  ----
  * 3상 전압 v1, v2, v3 는 같은 60Hz 와 고조파 구조를 공유하므로 이상적인
    경우 lag=0 에서 매우 높은 ρ 값이 나온다.
  * 만약 두 채널의 고조파 위치가 ΔHz 만큼 다르다면 (예: 회전수가 다른
    두 모터의 비교), ρ 의 피크가 ΔHz 만큼 옮겨 나타난다.
  * 따라서 ρ 의 피크 lag 위치는 "두 신호 사이의 평균적인 주파수 차이"
    를 직접 보여준다.

  구현
  ----
  * 입력 fft_mag : (6, n_win, n_bins) 단측 진폭 스펙트럼.
  * 각 윈도우 안에서 두 스펙트럼에서 평균을 빼고 분산으로 정규화한 뒤
    ``scipy.signal.correlate`` 로 full cross-correlation 을 구한다.
    그 중 ±max_lag bin 만 잘라 결과로 둔다.
  * 결과 행렬의 lag 축 단위는 ``bin`` 이다. 실제 Hz 로 환산하려면
    ``lag_hz = lag_bin * (fs / samples_per_window)`` 를 쓰면 된다.
  """
  name_to_idx = {n: i for i, n in enumerate(CHANNEL_NAMES)}
  n_chan, n_win, n_bins = fft_mag.shape
  lags = np.arange(-max_lag, max_lag + 1)
  # 시간영역과 동일한 규약: full 결과의 중앙 인덱스가 lag=0.
  center = n_bins - 1

  result = {}
  for a_name, b_name in XCORR_PAIRS:
    ia = name_to_idx[a_name]
    ib = name_to_idx[b_name]
    out = np.empty((n_win, lags.size), dtype=np.float64)
    for w in range(n_win):
      xa = fft_mag[ia, w].astype(np.float64).copy()
      xb = fft_mag[ib, w].astype(np.float64).copy()
      xa -= xa.mean()
      xb -= xb.mean()
      denom = np.sqrt((xa * xa).sum() * (xb * xb).sum())
      if denom < 1e-12:
        out[w] = 0.0
        continue
      c = correlate(xa, xb, mode="full", method="fft") / denom
      out[w] = c[center - max_lag : center + max_lag + 1]
    result[f"{a_name}-{b_name}"] = out

  return result, lags


def _write_fft(h5file, fft_mag, freqs, samples_per_window, fs):
  """FFT 진폭 스펙트럼을 h5 그룹에 저장.

  그룹 구조 (group_name = "fft_scaled"):
    /fft_scaled
      attrs:
        samples_per_window (int) : 한 청크의 샘플 수
        fs_hz (int)              : 샘플링 주파수
        window (S)               : "hann"
        normalization (S)        : "amplitude_single_sided"
        source (S)               : "scaled"
      datasets:
        freqs : (n_bins,) 주파수 축 [Hz]
        v1    : (n_win, n_bins) 단측 진폭 스펙트럼
        v2, v3, i1, i2, i3 : 동일
  """
  if FFT_GROUP_SCALED in h5file:
    del h5file[FFT_GROUP_SCALED]
  g = h5file.create_group(FFT_GROUP_SCALED)
  g.attrs["samples_per_window"] = samples_per_window
  g.attrs["fs_hz"] = fs
  g.attrs["window"] = np.bytes_(b"hann")
  g.attrs["normalization"] = np.bytes_(b"amplitude_single_sided")
  g.attrs["source"] = np.bytes_(b"scaled")
  g.create_dataset("freqs", data=freqs)
  for i, ch in enumerate(CHANNEL_NAMES):
    g.create_dataset(ch, data=fft_mag[i], compression="gzip")


def _write_fft_xcorr(h5file, xcorr_dict, lags, samples_per_window, max_lag):
  """FFT 결과끼리의 cross-correlation 을 h5 그룹에 저장.

  그룹 구조 (group_name = "xcorr_fft_scaled"):
    /xcorr_fft_scaled
      attrs:
        max_lag (int)            : 저장한 lag 의 최대 절댓값(bin)
        samples_per_window (int) : FFT 1청크의 샘플 수
        pairs (S, ...)           : 채널 쌍 이름 목록
        source (S)               : "scaled"
        lag_unit (S)             : "bin"   (fs/samples_per_window Hz/bin)
      datasets:
        lags  : (2·max_lag + 1,) bin 단위 lag 축
        v1-v2 : (n_win, 2·max_lag + 1) 정규 cross-corr
        v2-v3 : ...
        ...
  """
  if XCORR_FFT_GROUP_SCALED in h5file:
    del h5file[XCORR_FFT_GROUP_SCALED]
  g = h5file.create_group(XCORR_FFT_GROUP_SCALED)
  g.attrs["max_lag"] = max_lag
  g.attrs["samples_per_window"] = samples_per_window
  g.attrs["pairs"] = np.array(list(xcorr_dict.keys()), dtype="S")
  g.attrs["source"] = np.bytes_(b"scaled")
  g.attrs["lag_unit"] = np.bytes_(b"bin")
  g.create_dataset("lags", data=lags)
  for name, arr in xcorr_dict.items():
    g.create_dataset(name, data=arr, compression="gzip")


def _compute_and_write_fft(h5file, scaled, fs,
                           fft_samples_per_window=FFT_WINDOW_SAMPLES,
                           max_lag=FFT_MAX_LAG_BINS):
  """실행사항 5 : scaled 데이터에 대해 FFT 와 FFT 결과쌍의 cross-correlation
  을 계산해 두 그룹 (``fft_scaled``, ``xcorr_fft_scaled``) 에 저장한다.
  """
  fft_mag, freqs = compute_fft_per_channel(scaled, fft_samples_per_window, fs)
  _write_fft(h5file, fft_mag, freqs, fft_samples_per_window, fs)
  xc_fft, lags = cross_correlate_fft_pairs(fft_mag, max_lag)
  _write_fft_xcorr(h5file, xc_fft, lags, fft_samples_per_window, max_lag)


def bundle(files, out_dir):
  raws = [load_phases(fp) for fp in files]
  raw = np.concatenate(raws, axis=1)  # (6, N)
  scaled = log16p(raw)  # (6, N)

  out_path = out_dir / f"{output_name(files[0])}.h5"
  with h5py.File(out_path, "w") as f:
    f.attrs["fs_hz"] = FS_HZ
    f.attrs["window_ms"] = WINDOW_MS
    f.attrs["samples_per_window"] = SAMPLES_PER_WINDOW
    f.attrs["channels"] = np.array(CHANNEL_NAMES, dtype="S")
    f.create_dataset("raw", data=raw, compression="gzip")
    f.create_dataset("scaled", data=scaled, compression="gzip")

    # ── 실행사항 3: 9개 채널쌍 cross-correlation 을 raw / scaled 양쪽에 대해
    # 계산해 동일 h5 의 별도 그룹에 저장 (xcorr_raw, xcorr_scaled).
    _compute_and_write_all_xcorr(
        f, raw, scaled, SAMPLES_PER_WINDOW, MAX_LAG_SAMPLES,
    )

    # ── 실행사항 5: scaled 데이터의 FFT (100,000 샘플 단위) 와 그 결과끼리의
    # cross-correlation 을 계산해 fft_scaled, xcorr_fft_scaled 그룹에 저장.
    _compute_and_write_fft(f, scaled, FS_HZ,
                           FFT_WINDOW_SAMPLES, FFT_MAX_LAG_BINS)


def add_xcorr_to_bundle(bundle_path, max_lag=MAX_LAG_SAMPLES,
                        fft_max_lag=FFT_MAX_LAG_BINS,
                        fft_samples_per_window=FFT_WINDOW_SAMPLES):
  """이미 만들어진 bundle h5 파일에 cross-correlation 및 FFT 관련 그룹을
  추가/갱신한다. bundle 을 다시 만들 필요 없이 raw, scaled 를 읽어
  cross-correlation 과 FFT 만 계산해 같은 파일에 덧붙인다.

  갱신/생성되는 그룹:
    * xcorr_raw, xcorr_scaled            (실행사항 3)
    * fft_scaled, xcorr_fft_scaled       (실행사항 5)
  """
  with h5py.File(bundle_path, "r+") as f:
    raw = f["raw"][:]
    scaled = f["scaled"][:]
    spw = int(f.attrs["samples_per_window"])
    fs_hz = int(f.attrs["fs_hz"])
    _compute_and_write_all_xcorr(f, raw, scaled, spw, max_lag)
    _compute_and_write_fft(f, scaled, fs_hz,
                           fft_samples_per_window, fft_max_lag)


def main():
  arg_path = sys.argv[1] if len(sys.argv) > 1 else "."
  base = Path(arg_path)
  files = sorted(
    (fp for fp in base.glob("motor_*.h5") if _NAME_RE.match(fp.name)),
    key=file_seq,
  )
  if files:
    bundle(files, base)
    return

  # raw 파일이 없으면, 이미 만들어진 bundle h5 에 xcorr 만 추가/갱신
  bundles = [fp for fp in base.glob("motor_*.h5") if not _NAME_RE.match(fp.name)]
  if not bundles:
    print("no motor_*.h5 files found")
    return
  for bp in bundles:
    add_xcorr_to_bundle(bp)
    print(f"updated xcorr in {bp}")

if __name__ == "__main__":
  main()
