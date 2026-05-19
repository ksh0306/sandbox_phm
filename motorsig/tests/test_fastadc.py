"""FastAdcData / concat_fast_adc 단위 테스트 (합성 데이터)."""

import numpy as np
import pytest

from motorsig import FastAdcData, concat_fast_adc

CH = ("v1", "v2", "v3", "i1", "i2", "i3")


def make_raw(n_packets=5, *, dtype=np.uint16, n_channels=6):
    """합성 fast_adc FastAdcData 생성."""
    rng = np.random.default_rng(n_packets)
    data = rng.integers(0, 60000, size=(n_packets, n_channels, 50))
    return FastAdcData(data.astype(dtype), CH[:n_channels], bits=16, fs=20000.0)


def test_2d_input_raises():
    with pytest.raises(ValueError):
        FastAdcData(np.zeros((6, 50), dtype=np.uint16), CH, bits=16)


def test_4d_input_raises():
    with pytest.raises(ValueError):
        FastAdcData(np.zeros((2, 6, 50, 1), dtype=np.uint16), CH, bits=16)


def test_channel_count_mismatch_raises():
    data = np.zeros((3, 4, 50), dtype=np.uint16)
    with pytest.raises(ValueError):
        FastAdcData(data, CH, bits=16)


def test_signed_dtype_raises():
    data = np.zeros((3, 6, 50), dtype=np.int16)
    with pytest.raises(ValueError):
        FastAdcData(data, CH, bits=16)


def test_float_dtype_raises():
    data = np.zeros((3, 6, 50), dtype=np.float64)
    with pytest.raises(ValueError):
        FastAdcData(data, CH, bits=16)


def test_concat_shape():
    merged = concat_fast_adc([make_raw(4), make_raw(7), make_raw(2)])
    assert merged.data.shape == (13, 6, 50)
    assert merged.channel_names == CH


def test_concat_empty_raises():
    with pytest.raises(ValueError):
        concat_fast_adc([])


def test_concat_channel_mismatch_raises():
    other = FastAdcData(
        np.zeros((3, 3, 50), dtype=np.uint16), CH[:3], bits=16
    )
    with pytest.raises(ValueError):
        concat_fast_adc([make_raw(4), other])


def test_concat_bits_mismatch_raises():
    other = FastAdcData(np.zeros((3, 6, 50), dtype=np.uint16), CH, bits=12)
    with pytest.raises(ValueError):
        concat_fast_adc([make_raw(4), other])


def test_concat_dtype_mismatch_raises():
    other = FastAdcData(np.zeros((3, 6, 50), dtype=np.uint8), CH, bits=16)
    with pytest.raises(ValueError):
        concat_fast_adc([make_raw(4), other])


def test_h5_roundtrip_bit_exact(tmp_path):
    original = make_raw(6)
    path = tmp_path / "raw.h5"
    original.to_h5(path)
    loaded = FastAdcData.from_h5(path)
    assert np.array_equal(loaded.data, original.data)
    assert loaded.data.dtype == original.data.dtype
    assert loaded.channel_names == original.channel_names
    assert loaded.bits == original.bits
    assert loaded.fs == original.fs


def test_from_h5_signed_reinterpreted_as_unsigned(tmp_path):
    """int16로 저장된 fast_adc는 비트패턴 유지한 채 uint16으로 재해석된다."""
    import h5py

    signed = np.array([[-32768, -1, 0, 1, 32767, 100]], dtype=np.int16)
    signed = np.tile(signed[:, :, None], (3, 1, 50))
    path = tmp_path / "signed.h5"
    with h5py.File(path, "w") as f:
        f.attrs["adc_effective_bits"] = 16
        f.attrs["fs_hz"] = 20000
        dset = f.create_dataset("fast_adc", data=signed)
        dset.attrs["channel_order"] = "v1,v2,v3,i1,i2,i3"
    loaded = FastAdcData.from_h5(path)
    assert loaded.data.dtype == np.uint16
    # 비트패턴 보존: int16 → uint16 view.
    assert np.array_equal(loaded.data, signed.view(np.uint16))
    assert loaded.bits == 16


def test_summary_keys():
    info = make_raw(5).summary()
    for key in ("shape", "dtype", "n_packets", "n_channels", "channel_names"):
        assert key in info
