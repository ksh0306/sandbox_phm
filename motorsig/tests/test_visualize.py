"""visualize 모듈 스모크 테스트 (합성 데이터, PNG 없이 Figure만 검증)."""

import matplotlib.pyplot as plt
import numpy as np

from motorsig import (
    CrossCorrFFT,
    CrossCorrLog,
    FastAdcData,
    FFTData,
    LogNormalized,
)
from motorsig.visualize import (
    plot_fft_bars,
    plot_waveforms,
    plot_xcorr_heatmaps,
    plot_xcorr_profiles,
    save_figure,
)

CH = ("v1", "v2", "v3", "i1", "i2", "i3")


def make_raw(n_packets=20):
    rng = np.random.default_rng(n_packets)
    data = rng.integers(0, 60000, size=(n_packets, 6, 50)).astype(np.uint16)
    return FastAdcData(data, CH, bits=16, fs=20000.0)


def test_plot_waveforms_returns_figure():
    raw = make_raw()
    norm = LogNormalized(raw)
    fig = plot_waveforms([raw, norm], labels=["raw", "norm"], length=200)
    assert len(fig.axes) == 12  # 6채널 × 2신호
    plt.close(fig)


def test_plot_fft_bars():
    fft = FFTData(LogNormalized(make_raw()), packets_per_group=5)
    fig = plot_fft_bars(fft, max_freq=5000.0, fundamental=120.0)
    assert len(fig.axes) == 6
    plt.close(fig)


def test_plot_xcorr_profiles_log():
    norm = LogNormalized(make_raw())
    xc = CrossCorrLog(norm, pairs=[("v1", "v2"), ("i1", "i2")], max_lag=50)
    fig = plot_xcorr_profiles(xc)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_xcorr_heatmaps_fft():
    fft = FFTData(LogNormalized(make_raw(24)), packets_per_group=6)
    xc = CrossCorrFFT(fft, pairs=[("v1", "v2"), ("v2", "v3")])
    fig = plot_xcorr_heatmaps(xc, freq_resolution=1.0)
    assert fig.axes
    plt.close(fig)


def test_save_figure_writes_png(tmp_path):
    fft = FFTData(LogNormalized(make_raw()), packets_per_group=10)
    path = tmp_path / "out.png"
    save_figure(plot_fft_bars(fft), path)
    assert path.exists() and path.stat().st_size > 0
