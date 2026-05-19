"""FFTData 단위 테스트 (합성 신호)."""

import numpy as np
import pytest

from motorsig import FastAdcData, FFTData, LogNormalized

CH = ("v1", "v2", "v3", "i1", "i2", "i3")


def make_norm(n_packets=20, n_per=50, *, signal=None):
    """합성 LogNormalized 생성. signal이 주어지면 모든 채널에 동일 주입."""
    n_ch = 6
    if signal is None:
        rng = np.random.default_rng(n_packets)
        flat = rng.standard_normal(n_packets * n_per)
    else:
        flat = np.asarray(signal, dtype=np.float64)
    per_ch = flat.reshape(n_packets, 1, n_per)
    data = np.tile(per_ch, (1, n_ch, 1))
    return LogNormalized._from_fields(data, CH, 16, 20000.0, "synthetic")


def test_unnormalized_input_raises():
    """미정규화 FastAdcData 입력 → ValueError."""
    raw = FastAdcData(np.ones((5, 6, 50), dtype=np.uint16), CH, bits=16)
    with pytest.raises(ValueError):
        FFTData(raw)


def test_packets_per_group_zero_single_group():
    """packets_per_group=0 → 그룹 1개."""
    fft = FFTData(make_norm(12), packets_per_group=0)
    assert fft.data.shape[0] == 1


def test_packets_per_group_ceil():
    """packets_per_group>0 → 그룹 수 = ceil(패킷/그룹크기)."""
    fft = FFTData(make_norm(10), packets_per_group=4)
    assert fft.data.shape[0] == 3  # ceil(10/4)


def test_storage_layout_list_channel_data():
    """저장 레이아웃 [리스트(그룹), 채널, 데이터]."""
    fft = FFTData(make_norm(10), packets_per_group=5)
    assert fft.data.ndim == 3
    assert fft.data.shape[1] == 6


def test_sinusoid_peaks_at_expected_bin():
    """단일 정현파 → 해당 주파수 빈에 피크."""
    n_packets, n_per = 20, 50
    length = n_packets * n_per
    cycles = 13
    t = np.arange(length)
    sig = np.sin(2 * np.pi * cycles * t / length)
    fft = FFTData(make_norm(n_packets, n_per, signal=sig), packets_per_group=0)
    for ch in range(6):
        assert int(np.argmax(fft.data[0, ch])) == cycles


def test_list_input_accumulates_groups():
    """list 입력 시 그룹이 리스트 축으로 누적된다."""
    fft = FFTData([make_norm(8), make_norm(8), make_norm(8)])
    assert fft.data.shape[0] == 3


def test_channels_are_individual_fields():
    """채널이 channels 딕셔너리에 채널별 개별 필드로 보유된다."""
    fft = FFTData(make_norm(10), packets_per_group=5)
    assert set(fft.channels) == set(CH)
    for name in CH:
        assert fft.channels[name].shape == (fft.data.shape[0], fft.data.shape[2])
        np.testing.assert_array_equal(
            fft.channels[name], fft.data[:, CH.index(name), :]
        )


def test_h5_stores_per_channel_datasets(tmp_path):
    """h5 저장 시 /spectrum 그룹 아래 채널별 데이터셋으로 저장된다."""
    import h5py

    fft = FFTData(make_norm(10), packets_per_group=5)
    path = tmp_path / "fft.h5"
    fft.to_h5(path)
    with h5py.File(path, "r") as f:
        assert set(f["spectrum"].keys()) == set(CH)


def test_h5_roundtrip(tmp_path):
    fft = FFTData(make_norm(10), packets_per_group=4)
    path = tmp_path / "fft.h5"
    fft.to_h5(path)
    loaded = FFTData.from_h5(path)
    np.testing.assert_allclose(loaded.data, fft.data, rtol=1e-12)
    np.testing.assert_allclose(loaded.freqs, fft.freqs, rtol=1e-12)
    assert loaded.channel_names == fft.channel_names
    for name in CH:
        np.testing.assert_allclose(
            loaded.channels[name], fft.channels[name], rtol=1e-12
        )
