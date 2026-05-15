import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
V_DIV = 26.45 # 분압하여 0~3.3V 핀의 전압에 곱해지는 분압비(87V max)
V_REF = 3.3 # PIN 전압 스케일 

CHANNEL_NAMES = ["va", "vb", "vc", "ia", "ib", "ic"]
FILES = sorted(Path(".").glob("motor_*.h5"))

def load(path):
    with h5py.File(path, "r") as f:
        fs_hz = int(f.attrs["fs_hz"])
        d = f["fast_adc"][:]
    d_f = d.astype(np.uint16).astype(np.float64)
    d_f[:, :3, :] = d_f[:, :3, :] / 65536.0 * V_REF * V_DIV
    d_f[:, 3:, :] = (d_f[:, 3:, :]- 32768.0)/65536.0 * V_REF * 100.0
    d_f[:, 3:, :] -= d_f[:, 3:, :].mean(axis=1, keepdims=True)
    per_pkt = d_f.transpose(1, 0, 2)
    return per_pkt.reshape(6, -1), per_pkt.mean(axis=2), fs_hz


def plot_grid(t, data, title, fname, lw=0.5):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(t, data[i], lw=lw)
        kind, unit = ("Voltage", "V") if i < 3 else ("Current", "A")
        ax.set_title(f"ch{i}: {CHANNEL_NAMES[i]} ({kind})")
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"saved: {fname}")


for fp in FILES:
    raw, mean, fs_hz = load(fp)
    fs_pkt = fs_hz / 50
    t_raw = np.arange(raw.shape[1]) / fs_hz
    t_pkt = np.arange(mean.shape[1]) / fs_pkt
    stem = fp.stem
    print(f"\n{fp.name}: {mean.shape[1]} packets ({raw.shape[1]} samples/ch)")
    print(f"raw fs = {fs_hz} Hz, packet-mean fs = {fs_pkt:.0f} Hz, duration = {t_raw[-1]:.2f} s")

    plot_grid(
        t_pkt, mean,
        f"{fp.name} — packet-mean (50:1 decimation, {fs_pkt:.0f}Hz) — full {t_pkt[-1]:.1f}s",
        f"fast_adc_sine_{stem}.png",
        lw=0.6,
    )

    zoom_a, zoom_b = int(1.0 * fs_pkt), int(1.1 * fs_pkt)
    if zoom_b <= mean.shape[1]:
        plot_grid(
            t_pkt[zoom_a:zoom_b] - t_pkt[zoom_a],
            mean[:, zoom_a:zoom_b],
            f"{fp.name} — packet-mean zoom (0.1 s window from t=1.0s)",
            f"fast_adc_sine_zoom_{stem}.png",
            lw=1.0,
        )
    else:
        print(f"skip zoom: {fp.name} too short ({t_pkt[-1]:.2f}s < 1.1s)")

    N_raw = int(0.2 * fs_hz)
    plot_grid(
        t_raw[:N_raw], raw[:, :N_raw],
        f"{fp.name} — raw fast_adc (first {N_raw / fs_hz * 1000:.0f} ms @ {fs_hz}Hz) — PWM-like",
        f"fast_adc_raw_{stem}.png",
        lw=0.5,
    )
