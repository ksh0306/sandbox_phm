"""motor_*.h5 입력 여러 개를 하나로 연결하고, raw/log16p 표현과 전체 신호의
rFFT magnitude, 그리고 v-v, i-i 상별 cross-correlation을 묶어 단일 h5에 저장.

FFT는 청크 분할 없이 전체 신호 1회 변환 (50ms 청크는 저주파 해상도 = 20Hz
부터만 잡혀서 저주파 성분이 사라짐). scaled_xcorr는 시변(時變) 추세를
보기 위해 시간 도메인 청크 단위로 계산.

implement.md 사양:
  - fast_adc: (n_packets, 6, 50) int16, 20kHz, 패킷 2.5ms 주기
  - 채널: v1,v2,v3,i1,i2,i3
  - 출력 파일명: 첫 입력 파일의 첫 3토큰(motor_id_date)까지

저장 데이터셋 (N = 전체 샘플 수, W = SAMPLES_PER_WINDOW = 1000):
  - raw           : (6, N) uint16            채널별 시계열
  - scaled        : (6, N) float64           log16p 정규화 (0~1)
  - raw_fft       : (6, N//2+1) float64      raw 전체 rFFT magnitude
  - scaled_fft    : (6, N//2+1) float64      scaled 전체 rFFT magnitude
  - fft_xcorr     : (6, 2(N//2+1)-1)         scaled_fft에서 v-v, i-i 페어 xcorr
  - scaled_xcorr  : (6, n_chunks, 2W-1)      scaled의 청크별 v-v, i-i 페어 xcorr
  - fft_freq      : (N//2+1,)                FFT bin 주파수 축 [Hz]
  - scaled_xcorr_lag_s  : (2W-1,)            시간 도메인 xcorr lag 축 [s]
  - fft_xcorr_lag_hz    : (2(N//2+1)-1,)     주파수 도메인 xcorr lag 축 [Hz]

xcorr 페어 (6개 채널):
  [v1-v2, v2-v3, v3-v1, i1-i2, i2-i3, i3-i1]
"""
from pathlib import Path
import re

import h5py
import numpy as np

CHANNEL_NAMES = ["v1", "v2", "v3", "i1", "i2", "i3"]
FS_HZ = 20000          # ADC 샘플링 [Hz]
WINDOW_MS = 50         # FFT 청크 길이 [ms]
SAMPLES_PER_WINDOW = FS_HZ * WINDOW_MS // 1000  # 1000

# v-v, i-i 페어 (채널 인덱스: 0=v1, 1=v2, 2=v3, 3=i1, 4=i2, 5=i3)
XCORR_PAIRS = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
XCORR_PAIR_NAMES = [f"{CHANNEL_NAMES[a]}-{CHANNEL_NAMES[b]}" for a, b in XCORR_PAIRS]


# ADC 입력 0~65535 → log2(x+1)/16 으로 0~1 범위 압축.
# +1은 log(0) = -inf 방지용. log2(65536)/16 = 1.0 이 상한.
def log16p(x):
    return np.log2(np.asarray(x, dtype=np.float64) + 1) / 16.0


def load_phases(path):
    """h5 파일에서 (6, N) 채널별 파형을 추출.
    fast_adc shape (n_packets, 6, 50) → transpose(1,0,2).reshape(6, -1).
    """
    with h5py.File(path, "r") as f:
        d = f["fast_adc"][:]
    return d.astype(np.uint16).transpose(1, 0, 2).reshape(6, -1)


def xcorr_per_chunk(chunked, pairs):
    """청크 단위 (C, n_chunks, L) 데이터에서 페어별 선형 cross-correlation.

    DSP 표준 관례:  r[m] = Σ_n a[n] · b[n+m]
      → 양의 lag m → b가 a보다 m 만큼 지연
      → 음의 lag m → b가 a보다 m 만큼 앞섬

    선형(비순환) xcorr를 위해 길이 2L로 zero-pad 후 F^-1[conj(A)·B] 계산.
    출력 lag 축은 -(L-1)..(L-1), 길이 2L-1. scipy.signal.correlate 관례와 동일.

    Returns: (n_pairs, n_chunks, 2L-1) float64
    """
    n_chans, n_chunks, L = chunked.shape
    n_pairs = len(pairs)
    if n_chunks == 0:
        return np.zeros((n_pairs, 0, 2 * L - 1), dtype=np.float64)
    n_pad = 2 * L
    X = np.fft.rfft(chunked.astype(np.float64), n=n_pad, axis=2)
    a_idx = np.array([p[0] for p in pairs])
    b_idx = np.array([p[1] for p in pairs])
    prod = np.conj(X[a_idx]) * X[b_idx]                 # (n_pairs, n_chunks, n_pad//2+1)
    cc = np.fft.irfft(prod, n=n_pad, axis=2)            # (n_pairs, n_chunks, n_pad)
    # 순환 인덱싱:
    #   양의 lag 0..L-1     → cc[..., 0..L-1]
    #   음의 lag -(L-1)..-1 → cc[..., L+1..2L-1]  (cc[..., L]은 미사용)
    neg = cc[:, :, L + 1:2 * L]
    pos = cc[:, :, :L]
    return np.concatenate([neg, pos], axis=2)


def chunked_xcorr_pairs(x, pairs, samples_per_window):
    """(C, N) 시계열을 청크로 분할한 뒤 페어별 청크 cross-correlation.

    Returns: (n_pairs, n_chunks, 2W-1) float64
    """
    n_chans, n = x.shape
    W = samples_per_window
    n_chunks = n // W
    if n_chunks == 0:
        return np.zeros((len(pairs), 0, 2 * W - 1), dtype=np.float64)
    usable = n_chunks * W
    chunked = x[:, :usable].reshape(n_chans, n_chunks, W)
    return xcorr_per_chunk(chunked, pairs)


def xcorr_pairs(x, pairs):
    """(C, L) 1D 시퀀스 묶음에서 페어별 선형 cross-correlation (단일 청크).

    xcorr_per_chunk를 singleton chunk 축으로 래핑. Returns: (n_pairs, 2L-1) float64.
    """
    return xcorr_per_chunk(x[:, None, :], pairs)[:, 0, :]


# motor_<id>_<date>_<seq>.h5 형태에서 첫 3토큰을 출력 이름으로 사용.
_NAME_RE = re.compile(r"^([^_]+_[^_]+_[^_]+)_.+\.h5$")
_SEQ_RE = re.compile(r"_(\d+)\.h5$")


def output_name(path):
    """입력 파일명에서 첫 3토큰(motor_id_date) 추출."""
    m = _NAME_RE.match(path.name)
    return m.group(1) if m else path.stem


def file_seq(path):
    """파일명 끝의 시퀀스 번호 (정렬용). 없으면 0."""
    m = _SEQ_RE.search(path.name)
    return int(m.group(1)) if m else 0


def bundle(files, out_dir):
    # 2. 채널별 reshape → 모든 입력 파일을 시간축으로 연결
    raws = [load_phases(fp) for fp in files]
    raw = np.concatenate(raws, axis=1)

    # 3. log16p 정규화
    scaled = log16p(raw)

    # 4. 전체 신호 1회 rFFT magnitude (저주파 손실 방지: 청크 분할 X)
    raw_fft = np.abs(np.fft.rfft(raw.astype(np.float64), axis=1))
    scaled_fft = np.abs(np.fft.rfft(scaled, axis=1))

    # 5. scaled_fft(=log16p 정규화 raw의 전체 FFT) 페어별 cross-correlation
    fft_xcorr = xcorr_pairs(scaled_fft, XCORR_PAIRS)

    # 6. scaled raw-data의 페어별 청크 cross-correlation (시간 도메인 lag, 시변 분석)
    scaled_xcorr = chunked_xcorr_pairs(scaled, XCORR_PAIRS, SAMPLES_PER_WINDOW)

    # 축 메타데이터
    W = SAMPLES_PER_WINDOW
    N = raw.shape[1]
    n_bins_full = N // 2 + 1
    fft_freq = np.fft.rfftfreq(N, d=1.0 / FS_HZ)                       # (N//2+1,)
    scaled_xcorr_lag_s = np.arange(-(W - 1), W) / FS_HZ                 # (2W-1,)
    df = FS_HZ / N                                                       # 전체 FFT bin 간격 [Hz]
    fft_xcorr_lag_hz = np.arange(-(n_bins_full - 1), n_bins_full) * df  # (2*n_bins_full - 1,)

    # 7. 첫 입력 파일의 첫 3토큰을 이름으로 단일 h5 저장
    out_path = out_dir / f"{output_name(files[0])}.h5"
    with h5py.File(out_path, "w") as f:
        f.attrs["fs_hz"] = FS_HZ
        f.attrs["window_ms"] = WINDOW_MS
        f.attrs["samples_per_window"] = SAMPLES_PER_WINDOW
        f.attrs["channels"] = np.array(CHANNEL_NAMES, dtype="S")
        f.attrs["xcorr_pairs"] = np.array(XCORR_PAIR_NAMES, dtype="S")
        f.attrs["source_files"] = np.array([fp.name for fp in files], dtype="S")
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("scaled", data=scaled, compression="gzip")
        f.create_dataset("raw_fft", data=raw_fft, compression="gzip")
        f.create_dataset("scaled_fft", data=scaled_fft, compression="gzip")
        f.create_dataset("fft_xcorr", data=fft_xcorr, compression="gzip")
        f.create_dataset("scaled_xcorr", data=scaled_xcorr, compression="gzip")
        f.create_dataset("fft_freq", data=fft_freq)
        f.create_dataset("scaled_xcorr_lag_s", data=scaled_xcorr_lag_s)
        f.create_dataset("fft_xcorr_lag_hz", data=fft_xcorr_lag_hz)

    print(
        f"saved: {out_path}  "
        f"(files={len(files)}, raw={raw.shape}, raw_fft={raw_fft.shape}, "
        f"fft_xcorr={fft_xcorr.shape}, scaled_xcorr={scaled_xcorr.shape})"
    )


def main():
    # 4토큰(motor_id_date_seq.h5) 형태만 입력으로 인정.
    # 재실행 시 본 스크립트의 출력(3토큰)은 자기 자신을 다시 입력으로 잡지 않도록 제외.
    files = sorted(print(Path("Motor1000_fingerLoad").absolute() )
        (fp for fp in Path(".").glob("motor_*.h5") if _NAME_RE.match(fp.name)),
        key=file_seq,
    )
    if not files:
        print("no motor_*.h5 files found")
        return
    bundle(files, Path("."))


if __name__ == "__main__":
    main()
