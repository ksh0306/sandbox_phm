import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

DEFAULT_FILE = Path("Motor1000_NoLoad/motor_1_20260511.h5")
DEFAULT_START = 300000
DEFAULT_WINDOW = 100
DEFAULT_OUT_DIR = Path("pics")
SPLINE_OVERSAMPLE = 20


def load_bundle(path):
  with h5py.File(path, "r") as f:
    raw = f["raw"][:]
    scaled = f["scaled"][:]
    fs_hz = int(f.attrs["fs_hz"])
    channels = [c.decode() if isinstance(c, bytes) else c
          for c in f.attrs["channels"]]
  return raw, scaled, fs_hz, channels


XCORR_SOURCES = ("raw", "scaled")  # preproc 이 만들어두는 두 그룹


def load_xcorr(path, source):
  """h5 파일에서 cross-correlation 그룹 하나를 읽어온다.

  Parameters
  ----------
  path : Path
    bundle h5 파일 경로.
  source : str
    "raw" 또는 "scaled". 그룹 이름은 ``xcorr_{source}`` 이다.

  Returns
  -------
  None        : 해당 그룹이 없는 경우
  (data, lags, samples_per_window) : 정상 로드 시
    data : dict[str, ndarray (n_win, n_lag)]
      채널 쌍 이름 → 정규 cross-correlation 행렬.
      preproc 에서 저장한 순서(XCORR_PAIRS) 가 그대로 유지된다.
    lags : ndarray (n_lag,)
      샘플 단위 lag 축. fs 로 나누면 초.
    samples_per_window : int
      한 윈도우의 샘플 수 (시간축 환산용).
  """
  group_name = f"xcorr_{source}"
  with h5py.File(path, "r") as f:
    if group_name not in f:
      return None
    g = f[group_name]
    pair_names = [p.decode() if isinstance(p, bytes) else p
            for p in g.attrs["pairs"]]
    spw = int(g.attrs["samples_per_window"])
    lags = g["lags"][:]
    data = {name: g[name][:] for name in pair_names}
  return data, lags, spw


def _draw_channel(ax, t, y, t_dense):
  ax.plot(t, y, "o", markersize=3, color="C0", label="sample")
  if len(t) >= 2:
    cs = CubicSpline(t, y)
    ax.plot(t_dense, cs(t_dense), "-", linewidth=0.9, color="C1",
        label="spline")
  ax.grid(True, alpha=0.3)


def plot_signals(raw, scaled, channels, fs_hz, start, window):
  end = min(start + window, raw.shape[1])
  raw_seg = raw[:, start:end].astype(np.float64)
  scaled_seg = scaled[:, start:end].astype(np.float64)
  t = np.arange(start, end) / fs_hz
  t_dense = np.linspace(t[0], t[-1], (len(t) - 1) * SPLINE_OVERSAMPLE + 1) \
    if len(t) >= 2 else t

  fig, axes = plt.subplots(6, 2, figsize=(16, 11), sharex=True)
  axes[0, 0].set_title("raw")
  axes[0, 1].set_title("scaled")
  for i in range(6):
    _draw_channel(axes[i, 0], t, raw_seg[i], t_dense)
    _draw_channel(axes[i, 1], t, scaled_seg[i], t_dense)
    axes[i, 0].set_ylabel(channels[i])
  axes[0, 0].legend(loc="upper right", fontsize=8)
  axes[-1, 0].set_xlabel("time [s]")
  axes[-1, 1].set_xlabel("time [s]")
  fig.suptitle(f"start={start}  window={end - start}")
  fig.tight_layout()
  return fig


# ─────────────────────────────────────────────────────────────────────────────
# 실행사항 4 : Cross-correlation 시각화
# ─────────────────────────────────────────────────────────────────────────────
# preproc 단계에서 만든 9개 쌍의 정규 cross-correlation 행렬을 사람이 보기
# 좋은 두 가지 형태로 표현한다.
#
#   (1) 히트맵 : (시간 윈도우, lag) 평면을 색으로 칠한 2D 지도.
#       - "시간이 흐르면서 두 신호 간의 위상관계가 어떻게 변하는가" 를 본다.
#       - 일정한 lag 위치에 안정적인 띠가 보이면, 그 시간 동안 두 신호가
#         특정 위상차로 강하게 결합되어 있다는 뜻이다.
#       - 띠가 좌우로 흔들리거나 끊기면, 모터 부하나 속도 변동, 노이즈 등
#         때문에 위상 관계가 흔들리고 있다는 뜻이다.
#
#   (2) 시간평균 프로파일 : 행렬을 시간축(윈도우 축) 으로 평균한 1D 곡선.
#       - "전체 측정 구간을 통틀어 봤을 때 두 신호의 평균적인 위상 관계는?"
#       - 피크의 lag 위치 = 두 신호 사이의 평균적인 시간 지연.
#         예: 60Hz 3상 신호라면 v1-v2 의 피크가 약 +1/3 주기 (≈ +5.6 ms)
#         부근에 있어야 한다 (이상적인 정현파일 때, 부호는 측정 정의에 따름).
#       - 곡선이 lag 축을 따라 거의 평탄하면 두 신호는 상관관계가 없다.
# ─────────────────────────────────────────────────────────────────────────────


def _xcorr_grid_shape(n_pairs):
  """채널쌍 개수에 맞는 서브플롯 행/열 수. 9 쌍이면 (3, 3)."""
  ncols = 3
  nrows = (n_pairs + ncols - 1) // ncols
  return nrows, ncols


def plot_xcorr_heatmaps(xcorr_data, lags, fs_hz, samples_per_window, source=""):
  """9개 채널쌍의 cross-correlation 행렬을 히트맵 그리드로 그린다.

  각 서브플롯 의미
  ----------------
  * x 축 : lag [ms]. 양수면 첫번째 채널(예: v1-v2 의 v1) 이 두번째 채널
       보다 그만큼 "지연(lag)" 되어 있다는 뜻 (preproc 의 부호 규약).
  * y 축 : time [s]. 윈도우 인덱스 × (samples_per_window / fs).
       y 가 커질수록 측정 후반부.
  * 색상 : 정규 cross-correlation 값 ρ ∈ [-1, +1].
       빨강(+1) = 강한 양의 상관, 파랑(-1) = 강한 음의 상관,
       하양(0) = 무상관.

  의미 있는 패턴 예시
  -------------------
  * 수직으로 늘어선 빨간 띠 : 시간이 흘러도 일정한 lag 에서 두 신호가
    강하게 닮음 → 위상 관계가 안정.
  * 수직 띠가 여러 줄로 주기적으로 반복 : 한 주기(60Hz 라면 ≈16.7 ms)
    간격으로 lag 가 반복된다는 의미. 정현파 두 개가 위상차를 두고
    있을 때 전형적으로 나타나는 모양이다.
  * 띠가 시간에 따라 좌우로 흔들림 : 회전수/부하가 변하면서 위상이 떠다님.
  * 띠가 갑자기 사라지거나 비대칭으로 깨짐 : 결함, 노이즈, 트랜지언트 등.
  """
  names = list(xcorr_data.keys())
  nrows, ncols = _xcorr_grid_shape(len(names))

  # constrained_layout 을 사용하면 colorbar 와 subplot 의 간격 조정을
  # matplotlib 가 알아서 해 준다 (tight_layout 보다 안정적).
  fig, axes = plt.subplots(
    nrows, ncols, figsize=(16, 11),
    constrained_layout=True,
  )
  axes_flat = np.atleast_1d(axes).flatten()

  # lag 축을 ms 로 환산. fs=20kHz 이면 1 샘플 = 0.05 ms.
  lag_ms = lags / fs_hz * 1000.0

  im = None
  for ax, name in zip(axes_flat, names):
    m = xcorr_data[name]                         # (n_win, n_lag)
    n_win = m.shape[0]
    # 시간축: 각 윈도우의 시작 시각 [s].
    t0 = 0.0
    t1 = n_win * samples_per_window / fs_hz

    # imshow 의 extent 로 축을 물리 단위(ms, s) 로 매핑.
    #   - origin='lower' : 행 0 이 그림 하단 → 시간 흐름이 위로 향한다.
    #   - aspect='auto'  : 가로/세로 비율을 데이터 비율 무시하고 채워 그림.
    #   - cmap='RdBu_r'  : 0 중심의 빨강-파랑 발산 컬러맵 (상관계수에 적합).
    #   - vmin/vmax=-1/+1 : 모든 서브플롯의 색상 스케일을 통일.
    im = ax.imshow(
      m,
      aspect="auto",
      origin="lower",
      extent=[lag_ms[0], lag_ms[-1], t0, t1],
      cmap="RdBu_r",
      vmin=-1.0,
      vmax=1.0,
      interpolation="nearest",
    )
    ax.set_title(name)
    ax.set_xlabel("lag [ms]")
    ax.set_ylabel("time [s]")
    # lag = 0 기준선을 얇게 그어 시각적인 기준점 제공
    ax.axvline(0.0, color="k", linewidth=0.5, alpha=0.4)

  # 사용하지 않는 자리는 숨김 (9 쌍을 3×3 으로 채우면 남는 자리가 없지만,
  # 추후 쌍 개수가 바뀌어도 안전하도록 처리)
  for ax in axes_flat[len(names):]:
    ax.axis("off")

  # 9 개 subplot 공통 colorbar (오른쪽). im 은 마지막 imshow 핸들이지만
  # vmin/vmax 가 모두 같으므로 색상 매핑이 일치한다.
  if im is not None:
    fig.colorbar(
      im,
      ax=axes_flat.tolist(),
      shrink=0.85,
      label="normalized cross-correlation ρ",
    )
  title = "Cross-correlation heatmaps (time × lag, ρ ∈ [-1, +1])"
  if source:
    title = f"[{source}] {title}"
  fig.suptitle(title)
  return fig


def plot_xcorr_profiles(xcorr_data, lags, fs_hz, source=""):
  """9개 채널쌍의 시간평균 cross-correlation 프로파일을 선 그래프로 그린다.

  각 서브플롯 의미
  ----------------
  * 입력 행렬 (n_win, n_lag) 을 윈도우 축(axis=0) 으로 평균하여 lag 축
    길이 1D 곡선 ρ̄(lag) 를 만든다.
  * x 축 : lag [ms].
  * y 축 : 평균 정규 cross-correlation. 이론적 범위 [-1, +1] 이지만,
       평균을 취하므로 실제로는 그보다 좁은 범위에 들어온다.

  읽는 법
  -------
  * 피크 위치 (절댓값 최대인 lag) = 두 신호의 평균적인 시간 지연.
  * 피크 부호 (양/음) = 동상/역상.
  * 곡선이 lag = 0 근처에서 좌우대칭적인 모양으로 0 중심으로 부드럽게
    떨어지면 두 신호가 거의 동일한 신호 + 노이즈 형태.
  * 사인꼴(주기적으로 좌우로 출렁) 모양이면 두 신호 모두 정현파 성분이
    강하며, 한 주기 간격(60Hz 라면 ≈16.7 ms) 으로 부호가 반전된다.
  """
  names = list(xcorr_data.keys())
  nrows, ncols = _xcorr_grid_shape(len(names))

  fig, axes = plt.subplots(
    nrows, ncols, figsize=(16, 9),
    sharex=True, sharey=True,
    constrained_layout=True,
  )
  axes_arr = np.atleast_2d(axes)
  axes_flat = axes_arr.flatten()

  lag_ms = lags / fs_hz * 1000.0

  for ax, name in zip(axes_flat, names):
    m = xcorr_data[name]                       # (n_win, n_lag)
    # 시간평균 프로파일: 모든 윈도우의 cross-corr 곡선을 평균.
    profile = m.mean(axis=0)

    ax.plot(lag_ms, profile, linewidth=1.0, color="C0")
    # 기준선: lag=0 수직선, ρ=0 수평선
    ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.4)
    ax.axvline(0.0, color="k", linewidth=0.5, alpha=0.4)

    # 절댓값 최대 위치를 피크로 표시 (양/음 상관 모두 잡기 위함)
    pk_idx = int(np.argmax(np.abs(profile)))
    ax.plot(lag_ms[pk_idx], profile[pk_idx], "ro", markersize=4)
    ax.set_title(
      f"{name}  peak @ {lag_ms[pk_idx]:+.2f} ms "
      f"(ρ̄={profile[pk_idx]:+.3f})"
    )
    ax.grid(True, alpha=0.3)

  for ax in axes_flat[len(names):]:
    ax.axis("off")

  # 공통 축 라벨은 가장자리 서브플롯에만 부여한다.
  for ax in axes_arr[-1, :]:
    ax.set_xlabel("lag [ms]")
  for ax in axes_arr[:, 0]:
    ax.set_ylabel("mean ρ")

  title = "Time-averaged cross-correlation profiles"
  if source:
    title = f"[{source}] {title}"
  fig.suptitle(title)
  return fig


def main():
  p = argparse.ArgumentParser(description="PHM h5 파형 시각화")
  p.add_argument("file", nargs="?", default=str(DEFAULT_FILE),
                 help="번들 h5 경로")
  p.add_argument("--start", type=int, default=DEFAULT_START,
                 help="시작 시점 [샘플]")
  p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                 help="주기 [샘플]")
  p.add_argument("--start-sec", type=float, default=None,
                 help="시작 시점 [초] (--start 보다 우선)")
  p.add_argument("--window-sec", type=float, default=None,
                 help="주기 [초] (--window 보다 우선)")
  p.add_argument("--out-dir", type=Path, default=None,
                 help="PNG 저장 디렉토리. 지정하지 않으면 pics/<input 파일의 부모 폴더명>/")
  p.add_argument("--skip-time", action="store_true",
                 help="시간영역 raw/scaled plot 을 생략")
  p.add_argument("--skip-xcorr", action="store_true",
                 help="cross-correlation 히트맵/프로파일 plot 을 생략")
  p.add_argument("--show", action="store_true",
                 help="저장 후 화면에 표시")
  args = p.parse_args()

  file_path = Path(args.file)
  raw, scaled, fs_hz, channels = load_bundle(file_path)

  start = int(args.start_sec * fs_hz) if args.start_sec is not None else args.start
  window = int(args.window_sec * fs_hz) if args.window_sec is not None else args.window

  if start < 0 or start >= raw.shape[1]:
    raise SystemExit(f"start out of range: {start} / {raw.shape[1]}")
  if window <= 0:
    raise SystemExit(f"window must be positive: {window}")

  out_dir = args.out_dir if args.out_dir is not None else (
    DEFAULT_OUT_DIR / file_path.parent.name
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  if not args.skip_time:
    fig = plot_signals(raw, scaled, channels, fs_hz, start, window)
    out_path = out_dir / f"{file_path.stem}_s{start}_w{window}.png"
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")

  if not args.skip_xcorr:
    # raw / scaled 두 그룹에 대해 각각 히트맵, 시간평균 프로파일 PNG 를
    # 생성한다. 총 4개의 PNG (또는 그룹이 없으면 그만큼 건너뜀).
    any_loaded = False
    for source in XCORR_SOURCES:
      xcorr = load_xcorr(file_path, source)
      if xcorr is None:
        continue
      any_loaded = True
      xc_data, xc_lags, xc_spw = xcorr

      fig_hm = plot_xcorr_heatmaps(
        xc_data, xc_lags, fs_hz, xc_spw, source=source,
      )
      hm_path = out_dir / f"{file_path.stem}_xcorr_{source}_heatmap.png"
      fig_hm.savefig(hm_path, dpi=120)
      print(f"saved {hm_path}")

      fig_pr = plot_xcorr_profiles(
        xc_data, xc_lags, fs_hz, source=source,
      )
      pr_path = out_dir / f"{file_path.stem}_xcorr_{source}_profile.png"
      fig_pr.savefig(pr_path, dpi=120)
      print(f"saved {pr_path}")

    if not any_loaded:
      print(f"[skip xcorr] '{file_path}' 에 xcorr_raw / xcorr_scaled "
          "그룹이 없습니다. phm_data_preproc.py 를 먼저 실행하세요.")

  if args.show:
    plt.show()


if __name__ == "__main__":
    main()
