"""CrossCorrLog / CrossCorrFFT 단위 테스트 (합성 신호)."""

import numpy as np
import pytest

from motorsig import CrossCorrFFT, CrossCorrLog, FastAdcData, FFTData, LogNormalized

CH = ("v1", "v2", "v3", "i1", "i2", "i3")


def make_norm_from_flat(flat_per_channel, n_packets, n_per):
    """채널별 1D 신호 리스트를 LogNormalized로 묶는다."""
    flat = np.stack(flat_per_channel, axis=0)  # (C, L)
    data = flat.reshape(6, n_packets, n_per).transpose(1, 0, 2)
    return LogNormalized._from_fields(data, CH, 16, 20000.0, "synthetic")


def test_shifted_pair_argmax_lag():
    """k 샘플 시프트된 채널쌍 → 시간영역 argmax lag == k.

    v1 = 기준 신호, v2 = v1을 k 샘플 지연시킨 신호.
    pair (v2, v1)의 상관 최댓값 lag는 지연량 k와 같다.
    """
    n_packets, n_per, k = 6, 50, 12
    length = n_packets * n_per
    rng = np.random.default_rng(0)
    base = rng.standard_normal(length)
    delayed = np.zeros(length)
    delayed[k:] = base[: length - k]
    zeros = np.zeros(length)
    norm = make_norm_from_flat(
        [base, delayed, zeros, zeros, zeros, zeros], n_packets, n_per
    )
    xc = CrossCorrLog(norm, pairs=[("v2", "v1")], max_lag=60)
    lag = int(xc.lags[np.argmax(xc.data[0, 0])])
    assert lag == k


def test_default_pairs_all_combinations():
    """pairs=None → 전체 채널 조합 C(6,2)=15."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((4, 6, 50))
    norm = LogNormalized._from_fields(data, CH, 16, 20000.0, "synthetic")
    xc = CrossCorrLog(norm)
    assert len(xc.pairs) == 15
    assert xc.data.shape[1] == 15


def test_crosscorrlog_rejects_fftdata():
    """CrossCorrLog 입력 타입 가드: LogNormalized만 허용."""
    rng = np.random.default_rng(2)
    norm = LogNormalized._from_fields(
        rng.standard_normal((4, 6, 50)), CH, 16, 20000.0, "x"
    )
    fft = FFTData(norm)
    with pytest.raises(ValueError):
        CrossCorrLog(fft)


def test_crosscorrlog_rejects_raw():
    raw = FastAdcData(np.ones((4, 6, 50), dtype=np.uint16), CH, bits=16)
    with pytest.raises(ValueError):
        CrossCorrLog(raw)


def test_crosscorrfft_rejects_lognorm():
    """CrossCorrFFT 입력 타입 가드: FFTData만 허용."""
    rng = np.random.default_rng(3)
    norm = LogNormalized._from_fields(
        rng.standard_normal((4, 6, 50)), CH, 16, 20000.0, "x"
    )
    with pytest.raises(ValueError):
        CrossCorrFFT(norm)


def test_crosscorrfft_shape():
    rng = np.random.default_rng(4)
    norm = LogNormalized._from_fields(
        rng.standard_normal((12, 6, 50)), CH, 16, 20000.0, "x"
    )
    fft = FFTData(norm, packets_per_group=4)  # 3 그룹
    xc = CrossCorrFFT(fft, pairs=[("v1", "v2"), ("i1", "i2")])
    assert xc.data.shape[0] == 3
    assert xc.data.shape[1] == 2


def test_pairs_are_individual_fields():
    """채널쌍이 pair_data 딕셔너리에 쌍별 개별 필드로 보유된다."""
    rng = np.random.default_rng(7)
    norm = LogNormalized._from_fields(
        rng.standard_normal((4, 6, 50)), CH, 16, 20000.0, "x"
    )
    pairs = [("v1", "v2"), ("i1", "i2"), ("v1", "i1")]
    xc = CrossCorrLog(norm, pairs=pairs, max_lag=30)
    assert set(xc.pair_data) == {"v1-v2", "i1-i2", "v1-i1"}
    for j, (a, b) in enumerate(pairs):
        np.testing.assert_array_equal(xc.pair_data[f"{a}-{b}"], xc.data[:, j, :])


def test_h5_stores_per_pair_datasets(tmp_path):
    """h5 저장 시 /xcorr 그룹 아래 채널쌍별 데이터셋으로 저장된다."""
    import h5py

    rng = np.random.default_rng(8)
    norm = LogNormalized._from_fields(
        rng.standard_normal((4, 6, 50)), CH, 16, 20000.0, "x"
    )
    xc = CrossCorrLog(norm, pairs=[("v1", "v2"), ("v2", "v3")], max_lag=30)
    path = tmp_path / "xc.h5"
    xc.to_h5(path)
    with h5py.File(path, "r") as f:
        assert set(f["xcorr"].keys()) == {"v1-v2", "v2-v3"}


def test_crosscorrlog_h5_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    norm = LogNormalized._from_fields(
        rng.standard_normal((4, 6, 50)), CH, 16, 20000.0, "x"
    )
    xc = CrossCorrLog(norm, pairs=[("v1", "v2")], max_lag=30)
    path = tmp_path / "xc.h5"
    xc.to_h5(path)
    loaded = CrossCorrLog.from_h5(path)
    np.testing.assert_allclose(loaded.data, xc.data, rtol=1e-12)
    np.testing.assert_array_equal(loaded.lags, xc.lags)
    assert loaded.pairs == xc.pairs


def test_crosscorrfft_h5_roundtrip(tmp_path):
    rng = np.random.default_rng(6)
    norm = LogNormalized._from_fields(
        rng.standard_normal((8, 6, 50)), CH, 16, 20000.0, "x"
    )
    fft = FFTData(norm, packets_per_group=4)
    xc = CrossCorrFFT(fft, pairs=[("v1", "v2")])
    path = tmp_path / "xcf.h5"
    xc.to_h5(path)
    loaded = CrossCorrFFT.from_h5(path)
    np.testing.assert_allclose(loaded.data, xc.data, rtol=1e-12)
    assert loaded.pairs == xc.pairs
