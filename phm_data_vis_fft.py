"""실행사항 6 : FFT 스펙트럼과 FFT 쌍의 cross-correlation 시각화.

phm_data_preproc.py 가 만들어 둔 두 그룹을 읽어 PNG 로 저장한다.

읽어들이는 h5 그룹
------------------
* /fft_scaled
    채널별 단측 진폭 스펙트럼 ``mag[ch][n_win, n_bins]`` + 주파수 축 ``freqs``.
    창함수(Hann), coherent gain 으로 정규화된 진폭 (60Hz 정현파 입력이라면
    60Hz bin 의 막대 높이가 그 진폭에 가깝게 나온다).
* /xcorr_fft_scaled
    9개 채널쌍의 FFT 결과끼리 cross-correlation. ``(n_win, 2·max_lag+1)``
    행렬, lag 축 단위는 bin (1 bin = fs/samples_per_window Hz).

만들어내는 PNG (모두 ``--out-dir`` 에 저장)
------------------------------------------
* ``<stem>_fft_bars.png``
    채널 6개의 윈도우 평균 진폭 스펙트럼을 막대그래프로 표현. 60Hz 와 그
    고조파가 한눈에 보이는 0–``--max-freq`` Hz 범위만 잘라 그린다.
* ``<stem>_xcorr_fft_profile.png``
    9개 쌍 각각의 윈도우-평균 정규 cross-correlation 곡선 (시간평균
    프로파일). 피크의 주파수 lag = 두 채널 스펙트럼의 평균 주파수 차이.
* ``<stem>_xcorr_fft_heatmap.png``
    같은 행렬을 ``(시간 윈도우 × lag)`` 2D 지도로 그린다. 데이터 현황
    (윈도우마다 닮음이 어떻게 변하는가) 을 한눈에 본다.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

CHANNEL_NAMES = ["v1", "v2", "v3", "i1", "i2", "i3"]
DEFAULT_FILE = Path("Motor1000_NoLoad/motor_1_20260511.h5")
DEFAULT_OUT_DIR = Path("pics")
DEFAULT_MAX_FREQ_HZ = 500.0   # 막대그래프에서 보여줄 최대 주파수 [Hz]

FFT_GROUP = "fft_scaled"
XCORR_FFT_GROUP = "xcorr_fft_scaled"


def load_fft(path):
  """h5 의 fft_scaled 그룹을 읽어 (mag, freqs, samples_per_window, fs_hz) 반환."""
  with h5py.File(path, "r") as f:
    if FFT_GROUP not in f:
      return None
    g = f[FFT_GROUP]
    spw = int(g.attrs["samples_per_window"])
    # fs 는 fft 그룹 또는 파일 루트 어느 쪽에서나 읽을 수 있다.
    if "fs_hz" in g.attrs:
      fs_hz = int(g.attrs["fs_hz"])
    else:
      fs_hz = int(f.attrs["fs_hz"])
    freqs = g["freqs"][:]
    mag = np.stack([g[ch][:] for ch in CHANNEL_NAMES], axis=0)
  return mag, freqs, spw, fs_hz


def load_xcorr_fft(path):
  """h5 의 xcorr_fft_scaled 그룹을 읽어 (data, lags, samples_per_window, fs_hz)."""
  with h5py.File(path, "r") as f:
    if XCORR_FFT_GROUP not in f:
      return None
    g = f[XCORR_FFT_GROUP]
    pair_names = [p.decode() if isinstance(p, bytes) else p
            for p in g.attrs["pairs"]]
    spw = int(g.attrs["samples_per_window"])
    lags = g["lags"][:]
    data = {name: g[name][:] for name in pair_names}
    fs_hz = int(f.attrs["fs_hz"])
  return data, lags, spw, fs_hz


# ─────────────────────────────────────────────────────────────────────────────
# 1) FFT 막대그래프
# ─────────────────────────────────────────────────────────────────────────────
def plot_fft_bars(fft_mag, freqs, max_freq=DEFAULT_MAX_FREQ_HZ, source="scaled"):
  """채널별 FFT 진폭 스펙트럼 (윈도우 평균) 을 막대그래프로 그린다.

  축 의미
  -------
  * x : 주파수 [Hz]. 0 ~ ``max_freq`` 범위만 잘라 보여준다. fft 가 다루는
      전 범위는 0 ~ fs/2 = 10 kHz 이지만, 60Hz 기본파와 그 저차 고조파
      (120, 180, 240, 300, …) 가 분석에 중요하므로 보통 0–500 Hz 정도면
      충분하다 (--max-freq 로 조절 가능).
  * y : 정규 진폭. coherent gain (Hann 창의 합) 으로 나누고 단측 보정 (*2)
      이 들어가 있어, 같은 진폭의 정현파라면 창의 길이/모양과 무관하게
      비슷한 높이의 막대를 갖는다. 따라서 막대 높이를 직접 진폭으로 해석
      할 수 있다. (DC 와 Nyquist 빈은 *2 보정에서 제외.)

  구현 메모
  --------
  * 모든 윈도우의 스펙트럼을 평균해 단일 "대표" 스펙트럼을 얻는다.
    윈도우마다 진폭이 흔들리더라도 평균하면 안정적인 피크 위치가 드러난다.
  * 막대 너비 ``width = df = freqs[1] - freqs[0]`` 를 정확히 한 bin
    간격으로 두어 막대 사이에 빈틈이나 겹침이 생기지 않게 한다.
  """
  avg = fft_mag.mean(axis=1)                       # (6, n_bins)

  mask = freqs <= max_freq
  f_shown = freqs[mask]
  df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

  fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
  for i, ch in enumerate(CHANNEL_NAMES):
    # 각 채널에 색을 분리해 한눈에 구분되도록 한다.
    axes[i].bar(f_shown, avg[i, mask], width=df,
                color=f"C{i}", edgecolor="none", align="center")
    axes[i].set_ylabel(f"{ch}  |X|")
    axes[i].grid(True, alpha=0.3, axis="y")
    # 60Hz 기본파와 그 고조파 위치에 얇은 보조선을 그어 해석을 돕는다.
    for harm in (60, 120, 180, 240, 300, 360, 420, 480):
      if harm <= max_freq:
        axes[i].axvline(harm, color="k", linewidth=0.4, alpha=0.15)
  axes[-1].set_xlabel("frequency [Hz]")
  fig.suptitle(
    f"[{source}] FFT amplitude spectrum (window-averaged, 0–{max_freq:.0f} Hz)"
  )
  fig.tight_layout()
  return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2) FFT cross-correlation 시각화
# ─────────────────────────────────────────────────────────────────────────────
# lag 축 단위
#   * preproc 단계에서 저장한 lag 는 "FFT bin" 단위이다.
#   * 1 bin = fs / samples_per_window  Hz   (= 0.2 Hz, fs=20kHz, N=100000)
#   * 따라서 lag_hz = lag_bin * (fs / samples_per_window) 로 환산하면
#     "두 스펙트럼이 얼마만큼 주파수가 어긋났을 때 가장 닮았는가" 의 단위를
#     Hz 로 직관적으로 읽을 수 있다.
# ─────────────────────────────────────────────────────────────────────────────


def _grid_shape(n):
  ncols = 3
  nrows = (n + ncols - 1) // ncols
  return nrows, ncols


def plot_xcorr_fft_profile(xcorr_data, lags, fs, samples_per_window,
                           source="scaled"):
  """9개 쌍의 시간평균 (윈도우 평균) FFT cross-correlation 프로파일.

  축 의미
  -------
  * x : 주파수 lag [Hz]. 양수면 두번째 채널의 스펙트럼을 그만큼 오른쪽
      (더 높은 주파수쪽) 으로 옮겼을 때 첫번째 채널과 닮는다는 뜻.
  * y : 윈도우-평균 정규 cross-correlation. [-1, +1] 범위지만 실제로는
      평균이라 그보다 좁다.

  읽는 법
  -------
  * 피크가 lag = 0 근처에서 매우 높음 → 두 채널이 동일한 주파수 구조를
    공유. 3상 전압 v1/v2/v3 처럼 같은 60Hz 와 고조파를 갖는 신호는 이런
    모양이 나오는 것이 정상이다.
  * 피크가 lag ≠ 0 → 두 채널의 고조파/공진 주파수가 평균적으로 그만큼
    어긋나 있음.
  * 곡선이 lag = 0 기준으로 좌우 대칭에 가까움 → 두 스펙트럼이 자기상관
    구조를 비슷하게 공유한다는 의미.
  """
  names = list(xcorr_data.keys())
  nrows, ncols = _grid_shape(len(names))

  fig, axes = plt.subplots(
    nrows, ncols, figsize=(16, 9),
    sharex=True, sharey=True,
    constrained_layout=True,
  )
  axes_arr = np.atleast_2d(axes)
  axes_flat = axes_arr.flatten()

  # bin → Hz 환산 계수
  freq_per_bin = fs / samples_per_window
  lag_hz = lags * freq_per_bin

  for ax, name in zip(axes_flat, names):
    m = xcorr_data[name]                          # (n_win, n_lag)
    profile = m.mean(axis=0)

    ax.plot(lag_hz, profile, color="C0", linewidth=1.0)
    ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.4)
    ax.axvline(0.0, color="k", linewidth=0.5, alpha=0.4)

    pk = int(np.argmax(np.abs(profile)))
    ax.plot(lag_hz[pk], profile[pk], "ro", markersize=4)
    ax.set_title(
      f"{name}  peak @ {lag_hz[pk]:+.2f} Hz "
      f"(ρ̄={profile[pk]:+.3f})"
    )
    ax.grid(True, alpha=0.3)

  for ax in axes_flat[len(names):]:
    ax.axis("off")

  for ax in axes_arr[-1, :]:
    ax.set_xlabel("freq-shift lag [Hz]")
  for ax in axes_arr[:, 0]:
    ax.set_ylabel("mean ρ")

  fig.suptitle(f"[{source}] Time-averaged FFT cross-correlation profiles")
  return fig


def plot_xcorr_fft_heatmap(xcorr_data, lags, fs, samples_per_window,
                           source="scaled"):
  """9개 쌍의 FFT cross-correlation 을 (윈도우 × lag) 히트맵으로 그린다.

  각 축
  -----
  * x : 주파수 lag [Hz]. lag = 0 에 검은색 얇은 기준선을 함께 그린다.
  * y : 측정 시각 [s]. (윈도우 인덱스) × (FFT 한 청크의 길이 = 5 s).
      y 가 커질수록 측정 후반.
  * 색 : 정규 cross-correlation ρ ∈ [-1, +1] (RdBu_r 컬러맵).

  의미
  ----
  "데이터 현황을 잘 볼 수 있는 시각화" 로서, 시간이 흐르며 두 신호의 주파수
  구조가 얼마나 안정적으로 같은지를 한 장에 보여준다. lag = 0 부근에
  빨간 띠가 시간축 전체에 안정적으로 깔려 있으면 두 채널의 스펙트럼이
  측정 내내 유사하다는 뜻이고, 띠가 끊기거나 lag 가 시간에 따라 이동하면
  부하/속도 변동이나 결함 발생 가능성을 시사한다.
  """
  names = list(xcorr_data.keys())
  nrows, ncols = _grid_shape(len(names))

  fig, axes = plt.subplots(
    nrows, ncols, figsize=(16, 11),
    constrained_layout=True,
  )
  axes_flat = np.atleast_1d(axes).flatten()

  freq_per_bin = fs / samples_per_window
  lag_hz = lags * freq_per_bin
  # 한 윈도우의 실제 길이 [초]. FFT_WINDOW_SAMPLES=100000, fs=20kHz → 5 s.
  win_dur_sec = samples_per_window / fs

  im = None
  for ax, name in zip(axes_flat, names):
    m = xcorr_data[name]
    n_win = m.shape[0]
    t0 = 0.0
    t1 = n_win * win_dur_sec
    im = ax.imshow(
      m,
      aspect="auto",
      origin="lower",
      extent=[lag_hz[0], lag_hz[-1], t0, t1],
      cmap="RdBu_r",
      vmin=-1.0,
      vmax=1.0,
      interpolation="nearest",
    )
    ax.set_title(name)
    ax.set_xlabel("freq-shift lag [Hz]")
    ax.set_ylabel("time [s]")
    ax.axvline(0.0, color="k", linewidth=0.5, alpha=0.4)

  for ax in axes_flat[len(names):]:
    ax.axis("off")

  if im is not None:
    fig.colorbar(
      im,
      ax=axes_flat.tolist(),
      shrink=0.85,
      label="normalized cross-correlation ρ",
    )

  fig.suptitle(
    f"[{source}] FFT cross-correlation heatmaps (time × freq-shift, "
    "ρ ∈ [-1, +1])"
  )
  return fig


def main():
  p = argparse.ArgumentParser(description="PHM h5 FFT 시각화")
  p.add_argument("file", nargs="?", default=str(DEFAULT_FILE),
                 help="bundle h5 경로 (fft_scaled / xcorr_fft_scaled 가 있어야 함)")
  p.add_argument("--out-dir", type=Path, default=None,
                 help="PNG 저장 디렉토리. 지정하지 않으면 pics/<input 파일의 부모 폴더명>/")
  p.add_argument("--max-freq", type=float, default=DEFAULT_MAX_FREQ_HZ,
                 help="FFT 막대그래프에 보여줄 최대 주파수 [Hz]")
  p.add_argument("--skip-fft", action="store_true",
                 help="FFT 막대그래프 생성을 생략")
  p.add_argument("--skip-xcorr", action="store_true",
                 help="FFT cross-correlation 그래프 생성을 생략")
  p.add_argument("--show", action="store_true",
                 help="저장 후 화면에 표시")
  args = p.parse_args()

  file_path = Path(args.file)
  out_dir = args.out_dir if args.out_dir is not None else (
    DEFAULT_OUT_DIR / file_path.parent.name
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  if not args.skip_fft:
    fft = load_fft(file_path)
    if fft is None:
      print(f"[skip fft] '{file_path}' 에 fft_scaled 그룹이 없습니다.")
    else:
      mag, freqs, spw, fs_hz = fft
      fig = plot_fft_bars(mag, freqs, max_freq=args.max_freq)
      out_path = out_dir / f"{file_path.stem}_fft_bars.png"
      fig.savefig(out_path, dpi=120)
      print(f"saved {out_path}")

  if not args.skip_xcorr:
    xc = load_xcorr_fft(file_path)
    if xc is None:
      print(f"[skip xcorr fft] '{file_path}' 에 xcorr_fft_scaled "
          "그룹이 없습니다.")
    else:
      xc_data, xc_lags, xc_spw, fs_hz = xc

      fig_pr = plot_xcorr_fft_profile(xc_data, xc_lags, fs_hz, xc_spw)
      pr_path = out_dir / f"{file_path.stem}_xcorr_fft_profile.png"
      fig_pr.savefig(pr_path, dpi=120)
      print(f"saved {pr_path}")

      fig_hm = plot_xcorr_fft_heatmap(xc_data, xc_lags, fs_hz, xc_spw)
      hm_path = out_dir / f"{file_path.stem}_xcorr_fft_heatmap.png"
      fig_hm.savefig(hm_path, dpi=120)
      print(f"saved {hm_path}")

  if args.show:
    plt.show()


if __name__ == "__main__":
    main()
