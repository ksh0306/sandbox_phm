"""LogNormalized / log16_plus1 단위 테스트 (명세 §3.3)."""

import numpy as np
import pytest

from motorsig import FastAdcData, LogNormalized, log16_plus1

CH = ("v1", "v2", "v3", "i1", "i2", "i3")


def make_raw(values, *, bits=16, dtype=np.uint16):
    """주어진 값으로 채워진 (1, 6, n) FastAdcData 생성."""
    arr = np.array(values, dtype=dtype)
    data = np.tile(arr[None, None, :], (1, 6, 1))
    return FastAdcData(data, CH, bits=bits, fs=20000.0)


def test_input_zero_maps_to_zero():
    """입력 0 → 출력 0 (log16(1) = 0)."""
    out = log16_plus1(np.array([0], dtype=np.uint16))
    assert out[0] == pytest.approx(0.0, abs=1e-9)


def test_12bit_boundary():
    """bits=12, 입력 4095 → 출력 ≈ 3.0."""
    out = log16_plus1(np.array([4095], dtype=np.uint16))
    assert out[0] == pytest.approx(3.0, abs=1e-6)


def test_16bit_boundary():
    """bits=16, 입력 65535 → 출력 ≈ 4.0."""
    out = log16_plus1(np.array([65535], dtype=np.uint16))
    assert out[0] == pytest.approx(4.0, abs=1e-6)


def test_overflow_uint16_max_not_broken():
    """uint16 최댓값 배열을 넣어도 음수/0으로 깨지지 않고 ≈ 4."""
    data = np.full((10,), 65535, dtype=np.uint16)
    out = log16_plus1(data)
    assert np.all(out > 3.9)
    assert np.all(out <= 4.0 + 1e-9)


def test_matches_naive_log16():
    """(b) 비트연산 구현이 naive np.log(X+1)/np.log(16)와 허용 오차 내 일치."""
    rng = np.random.default_rng(42)
    x = rng.integers(0, 65535, size=5000, endpoint=True).astype(np.uint16)
    fast = log16_plus1(x)
    naive = np.log(x.astype(np.float64) + 1.0) / np.log(16.0)
    np.testing.assert_allclose(fast, naive, rtol=1e-12, atol=1e-12)


def test_log16_plus1_rejects_signed():
    with pytest.raises(ValueError):
        log16_plus1(np.array([1, 2, 3], dtype=np.int32))


def test_is_a_fastadc():
    """IS-A 검증: LogNormalized 인스턴스는 FastAdcData이다."""
    norm = LogNormalized(make_raw([0, 100, 65535]))
    assert isinstance(norm, FastAdcData)


def test_source_unchanged():
    """정규화 후 입력 FastAdcData.data가 변하지 않는다 (원본 불변)."""
    raw = make_raw([0, 1, 2, 3, 4095, 65535])
    before = raw.data.copy()
    LogNormalized(raw)
    assert np.array_equal(raw.data, before)
    assert raw.data.dtype == np.uint16


def test_normalized_dtype_is_float():
    norm = LogNormalized(make_raw([1, 2, 3]))
    assert np.issubdtype(norm.data.dtype, np.floating)


def test_double_normalize_rejected():
    norm = LogNormalized(make_raw([1, 2, 3]))
    with pytest.raises(TypeError):
        LogNormalized(norm)


def test_h5_roundtrip(tmp_path):
    """h5 라운드트립 (부동소수점 허용 오차)."""
    norm = LogNormalized(make_raw([0, 7, 255, 4095, 65535]))
    path = tmp_path / "norm.h5"
    norm.to_h5(path)
    loaded = LogNormalized.from_h5(path)
    assert isinstance(loaded, LogNormalized)
    np.testing.assert_allclose(loaded.data, norm.data, rtol=1e-12)
    assert loaded.channel_names == norm.channel_names
    assert loaded.bits == norm.bits
