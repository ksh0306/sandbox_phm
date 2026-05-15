"""전압/전류 3상 파형의 raw, log16, FFT 표현을 추출하고 비교 시각화."""
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANNEL_NAMES = ["va", "vb", "vc", "ia", "ib", "ic"]
LOG16_EPSILON = 1e-9


def log16(x):
    return np.log2(np.asarray(x, dtype=np.float64) + LOG16_EPSILON) / 16.0


def extract_phases(path):
    """h5 파일에서 전압3상·전류3상 파형의 raw / log16 표현을 추출.

    FFT는 plot에 표시되는 윈도우 범위에 맞춰 plot_compare에서 계산한다.

    Returns dict:
        raw           : (6, N) uint16 ADC raw 값 (va, vb, vc, ia, ib, ic)
        scaled        : (6, N) log16(raw) 값
        t             : (N,) 시간축 [s] (ts_us 기반 정확한 샘플 간격)
        fs_hz         : 평균 샘플링 주파수 [Hz] (ts_us 로부터 산출)
        fs_hz_nominal : h5 attribute fs_hz [Hz]
        f_elec_hz     : slow_ctx.erpm 중앙값으로 산출한 모터 전기 기본주파수 [Hz]
        erpm_median   : slow_ctx.erpm 중앙값
        channels      : 채널 이름 리스트
        name          : 파일명 stem
    """
    # 입력 경로를 Path 객체로 정규화 (str/Path 둘 다 받을 수 있도록)
    path = Path(path)
    # h5 파일을 읽기 전용으로 열고 with 블록 내에서만 핸들 유지 (자원 자동 해제)
    with h5py.File(path, "r") as f:
        # 파일 attribute에 기록된 "공칭" 샘플링 주파수 (예: 10000 Hz). 펌웨어가 의도한 값.
        fs_hz_nominal = int(f.attrs["fs_hz"])
        # 한 패킷당 샘플 수. attribute가 없으면 기본 50으로 가정.
        samples_per_pkt = int(f.attrs.get("samples_per_pkt", 50))
        # fast_adc 데이터셋 통째로 메모리에 로드. shape = (n_packets, 6채널, 50샘플), int16.
        d = f["fast_adc"][:]  # (n_packets, 6, 50) int16
        # 각 패킷의 타임스탬프(마이크로초). 샘플 간격 산출에 사용. int64로 승격해 diff 오버플로 방지.
        ts_us = f["ts_us"][:].astype(np.int64)
        # 저속 컨텍스트(예: erpm 등 모터 상태)를 통째로 읽음. 구조화 배열.
        sc = f["slow_ctx"][:]
    # (패킷, 채널, 샘플) → (채널, 패킷, 샘플) 로 축 교환 후 (6, n_packets*50)으로 평탄화.
    # 결과적으로 채널별 시계열 한 줄로 이어붙인 형태가 된다. uint16으로 캐스팅 (ADC raw가 부호 없음).
    raw = d.astype(np.uint16).transpose(1, 0, 2).reshape(6, -1)

    # ─── 실측 ts_us 기반 샘플링 주파수 보정 ────────────────────────────
    # 우선 공칭값으로 초기화. 타임스탬프가 부족하면 그대로 사용.
    fs_hz = float(fs_hz_nominal)
    if len(ts_us) >= 2:
        # 인접 패킷 간 시간 간격 [us]. 길이는 len(ts_us)-1.
        diffs = np.diff(ts_us)
        # 공칭 fs로 계산한 "한 패킷이 차지해야 할 시간"[us].
        # 예) fs=10000Hz, samples_per_pkt=50 → 한 패킷 = 50/10000s = 5000us.
        nominal_pkt_us = samples_per_pkt * 1e6 / fs_hz
        # 패킷 간 idle gap을 제외하기 위해 nominal의 ±50% 범위만 평균
        # 데이터 수집이 잠시 멈췄다 재개되면 diffs에 거대한 값이 섞여 평균을 왜곡시키므로,
        # "정상적으로 연속된" 간격(공칭의 0.5배 ~ 1.5배 범위 내)만 골라낸다.
        consec = diffs[(diffs > 0.5 * nominal_pkt_us) & (diffs < 1.5 * nominal_pkt_us)]
        if len(consec) > 0:
            # 연속 패킷 간격의 평균[us]을 초로 환산 → 그것으로 samples_per_pkt를 나누면
            # 실제 샘플링 주파수[Hz]가 나온다. (샘플수 / 시간 = 주파수)
            # 이렇게 해서 펌웨어 클럭 오차/지터를 반영한 "실측" fs를 얻는다.
            fs_hz = samples_per_pkt / (float(consec.mean()) / 1e6)

    # ─── ERPM 기반 모터 전기 기본주파수 산출 ────────────────────────────
    # slow_ctx가 비어있을 수 있으므로 안전하게 가드. erpm 컬럼만 추출.
    erpm = sc["erpm"] if len(sc) else np.array([], dtype=np.int32)
    # 정지 구간(erpm<=0)은 기본주파수 계산에서 제외. 회전 중인 샘플만 사용.
    erpm_active = erpm[erpm > 0]
    # 평균 대신 중앙값 사용: 가속/감속 등 outlier에 덜 민감.
    erpm_median = float(np.median(erpm_active)) if len(erpm_active) else 0.0
    # erpm은 "전기적 RPM" → 60으로 나누면 전기 기본주파수[Hz].
    # (기계 RPM이 아니라 이미 극쌍수가 곱해진 전기 RPM이므로 별도 보정 불필요.)
    # 회전이 없으면 None을 반환해 호출자가 처리하도록 함.
    f_elec_hz = erpm_median / 60.0 if erpm_median > 0 else None

    # raw ADC를 log16 압축. 큰 동적 범위(0~65535)를 좁은 범위로 매핑하여 시각화/FFT에 유리.
    scaled = log16(raw)

    # 시간축[s]: 샘플 인덱스 / 샘플링 주파수. plot의 x축으로 사용.
    t = np.arange(raw.shape[1]) / fs_hz
    return {
        "raw": raw,
        "scaled": scaled,
        "t": t,
        "fs_hz": fs_hz,
        "fs_hz_nominal": fs_hz_nominal,
        "f_elec_hz": f_elec_hz,
        "erpm_median": erpm_median,
        "channels": list(CHANNEL_NAMES),
        "name": path.stem,
    }


def plot_compare(data, out_path, n_cycles=5, fundamental_hz=None,
                 fft_harmonics=20):
    """raw / raw-FFT / log16 / log16-FFT 네 표현을 6채널×4컬럼 그리드로 비교 시각화.

    n_cycles 개수만큼의 사이클만 시간축으로 표시하며, FFT 입력도 동일 윈도우로 한정한다.
    fundamental_hz가 None이면 extract_phases가 slow_ctx.erpm에서 산출한 f_elec_hz를 사용한다.
    """
    raw = data["raw"]
    scaled = data["scaled"]
    t = data["t"]
    fs_hz = data["fs_hz"]
    channels = data["channels"]

    if fundamental_hz is None:
        fundamental_hz = data.get("f_elec_hz")

    if fundamental_hz and fundamental_hz > 0:
        window_s = n_cycles / fundamental_hz
        title_window = f"{n_cycles} cycles @ {fundamental_hz:.1f} Hz ({window_s*1000:.1f} ms)"
        fft_xlim_hz = min(fft_harmonics * fundamental_hz, fs_hz / 2.0)
    else:
        window_s = 0.1
        title_window = f"first {window_s*1000:.0f} ms (fundamental N/A)"
        fft_xlim_hz = fs_hz / 2.0

    n_show = min(int(window_s * fs_hz), raw.shape[1])
    if n_show < 8:
        n_show = min(raw.shape[1], int(0.05 * fs_hz))

    # plot 윈도우 범위로 한정한 rFFT (raw, scaled 둘 다)
    raw_window = raw[:, :n_show].astype(np.float64)
    scaled_window = scaled[:, :n_show]
    fft_freq = np.fft.rfftfreq(n_show, d=1.0 / fs_hz)
    raw_fft_mag = np.abs(np.fft.rfft(raw_window, axis=1))
    scaled_fft_mag = np.abs(np.fft.rfft(scaled_window, axis=1))

    fig, axes = plt.subplots(6, 4, figsize=(20, 14), constrained_layout=True)
    for i in range(6):
        kind = "Voltage" if i < 3 else "Current"
        ax_raw, ax_raw_fft, ax_log, ax_log_fft = axes[i]

        ax_raw.plot(t[:n_show], raw[i, :n_show], lw=0.6)
        ax_raw.set_ylabel(f"{channels[i]}\n({kind})")
        ax_raw.grid(True, alpha=0.3)
        if i == 0:
            ax_raw.set_title(f"Raw ADC (uint16) — {title_window}")

        ax_raw_fft.plot(fft_freq, raw_fft_mag[i], lw=0.6, color="tab:blue")
        ax_raw_fft.set_xlim(0, fft_xlim_hz)
        ax_raw_fft.set_yscale("log")
        ax_raw_fft.grid(True, which="both", alpha=0.3)
        if fundamental_hz:
            ax_raw_fft.axvline(fundamental_hz, color="k", lw=0.5, ls="--", alpha=0.4)
        if i == 0:
            ax_raw_fft.set_title(f"FFT of Raw (≤ {fft_xlim_hz:.0f} Hz)")

        ax_log.plot(t[:n_show], scaled[i, :n_show], lw=0.6, color="tab:orange")
        ax_log.grid(True, alpha=0.3)
        if i == 0:
            ax_log.set_title(f"log16(Raw) — {title_window}")

        ax_log_fft.plot(fft_freq, scaled_fft_mag[i], lw=0.6, color="tab:red")
        ax_log_fft.set_xlim(0, fft_xlim_hz)
        ax_log_fft.set_yscale("log")
        ax_log_fft.grid(True, which="both", alpha=0.3)
        if fundamental_hz:
            ax_log_fft.axvline(fundamental_hz, color="k", lw=0.5, ls="--", alpha=0.4)
        if i == 0:
            ax_log_fft.set_title(f"FFT of log16 (≤ {fft_xlim_hz:.0f} Hz)")

    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("frequency [Hz]")
    axes[-1, 2].set_xlabel("time [s]")
    axes[-1, 3].set_xlabel("frequency [Hz]")

    f_str = f"{fundamental_hz:.2f} Hz" if fundamental_hz else "n/a"
    fig.suptitle(
        f"{data['name']}  (fs={fs_hz} Hz, N={raw.shape[1]} samples, "
        f"fft_window={n_show} samples, fundamental≈{f_str})",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved: {out_path}  (window={n_show} samples, f0={f_str})")


def main():
    files = sorted(Path(".").glob("motor_*.h5"))
    if not files:
        print("no motor_*.h5 files found")
        return
    for fp in files:
        data = extract_phases(fp)
        print(
            f"{fp.name}: fs={data['fs_hz']} Hz, "
            f"N={data['raw'].shape[1]} samples, "
            f"duration={data['t'][-1]:.2f} s"
        )
        plot_compare(data, f"phm_compare_{fp.stem}.png",3)


if __name__ == "__main__":
    main()
