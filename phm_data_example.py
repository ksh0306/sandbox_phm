"""클래스 기반 phm_data 파이프라인 예제.

arrangement.md 의 세 체크포인트마다 결과 파일을 만들고, 핵심 메타데이터를
출력해 sanity check 한다.

실행:
  uv run python phm_data_example.py [motor_folder]

기본 motor_folder: Motor1000_NoLoad/.
출력: _arrangement_out/ 하위에 8개의 h5 파일이 저장된다.

  체크포인트 1 (작업 0–3) : raw.h5, scaled.h5
  체크포인트 2 (작업 4–5) : fft_raw.h5, fft_scaled.h5
  체크포인트 3 (작업 6–8) : xcorr_time_raw.h5, xcorr_time_scaled.h5,
                            xcorr_fft_raw.h5,  xcorr_fft_scaled.h5
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import h5py

from phm_data import (
    FFTData,
    RawData,
    XcorrFFTData,
    XcorrTimeData,
    _NAME_RE,
    log16p,
)


OUT_DIR = Path("_arrangement_out")


# ─────────────────────────────────────────────────────────────────────────────
# 공통: 저장 직후 파일 요약 출력
# ─────────────────────────────────────────────────────────────────────────────
def _short_attr(v):
  """긴 ndarray attribute 는 잘라서 짧게 표시."""
  import numpy as np
  if isinstance(v, np.ndarray):
    if v.size > 6:
      return f"<ndarray dtype={v.dtype} shape={v.shape}>"
    return v.tolist()
  return v


def _report(path: Path) -> None:
  size = path.stat().st_size
  with h5py.File(path, "r") as f:
    print(f"  → {path}  ({size:,} bytes)")
    print(f"     attrs:")
    for k in sorted(f.attrs):
      print(f"       {k:<22s} = {_short_attr(f.attrs[k])}")
    print(f"     datasets:")
    for name in sorted(f.keys(), key=lambda n: (not n.isdigit(), n)):
      obj = f[name]
      if isinstance(obj, h5py.Dataset):
        print(f"       /{name:<20s} shape={obj.shape}, dtype={obj.dtype}")


# ─────────────────────────────────────────────────────────────────────────────
# 체크포인트 1 — 작업 0–3 : RawData + log16p
# ─────────────────────────────────────────────────────────────────────────────
def checkpoint1(src_dir: Path) -> tuple[Path, Path]:
  print(f"\n[CP1] raw chunk → RawData → log16p  (src={src_dir})")
  files = sorted(
    (fp for fp in src_dir.glob("motor_*.h5") if _NAME_RE.match(fp.name)),
    key=RawData.file_seq,
  )
  if not files:
    raise FileNotFoundError(f"raw chunk 파일을 찾을 수 없음: {src_dir}")
  print(f"  raw chunks: {len(files)}개")
  for fp in files:
    print(f"    - {fp.name}")

  # 0+1+2: raw chunk → RawData. fast_adc 외에도 events/fast_flags/
  # fast_motor_state/lifetime/motor_spec/slow_ctx/ts_us 까지 함께 보존된다.
  raw = RawData.from_chunks(files)
  print(f"  RawData  data.shape={raw.data.shape}, dtype={raw.data.dtype}, source={raw.source!r}")
  meta_keys = sorted(k for k in raw.datasets.keys() if k != "data")
  print(f"  meta datasets: {meta_keys}")
  raw_path = OUT_DIR / "raw.h5"
  raw.save(raw_path)
  _report(raw_path)

  # 3: log16p — data 만 변환하고 부수 메타데이터는 그대로 통과.
  scaled = log16p(raw)
  print(f"  log16p   data.shape={scaled.data.shape}, dtype={scaled.data.dtype}, "
        f"source={scaled.source!r}, min/max=({scaled.data.min():.4f}, {scaled.data.max():.4f})")
  scaled_path = OUT_DIR / "scaled.h5"
  scaled.save(scaled_path)
  _report(scaled_path)
  return raw_path, scaled_path


# ─────────────────────────────────────────────────────────────────────────────
# 체크포인트 2 — 작업 4–5 : FFTData.from_raw  (raw 와 scaled 양쪽)
# ─────────────────────────────────────────────────────────────────────────────
def checkpoint2(raw_path: Path, scaled_path: Path) -> tuple[Path, Path]:
  print(f"\n[CP2] FFTData.from_raw  (raw, scaled 양쪽)")

  for label, src in (("raw", raw_path), ("scaled", scaled_path)):
    print(f"\n  -- {label} 입력 ({src}) --")
    rd = RawData.load(src)
    fft = FFTData.from_raw(rd, samples_per_window=100_000)
    out = OUT_DIR / f"fft_{label}.h5"
    fft.save(out)
    print(f"  FFTData  fft_mag.shape={fft.fft_mag.shape}, "
          f"freqs[Δ]={fft.freqs[1] - fft.freqs[0]:.2f} Hz, source={fft.attrs['source']!r}")
    _report(out)

  return OUT_DIR / "fft_raw.h5", OUT_DIR / "fft_scaled.h5"


# ─────────────────────────────────────────────────────────────────────────────
# 체크포인트 3 — 작업 6–8 : XcorrTimeData / XcorrFFTData  (raw 와 scaled 양쪽)
# ─────────────────────────────────────────────────────────────────────────────
def checkpoint3(raw_path: Path, scaled_path: Path,
                fft_raw_path: Path, fft_scaled_path: Path) -> None:
  import numpy as np
  print(f"\n[CP3] xcorr 변환 4종 (시간영역 raw/scaled, FFT raw/scaled)")

  # 시간영역 xcorr — RawData 두 종 각각.
  for label, src in (("raw", raw_path), ("scaled", scaled_path)):
    rd = RawData.load(src)
    xc = XcorrTimeData.from_signal(rd)
    out = OUT_DIR / f"xcorr_time_{label}.h5"
    xc.save(out)
    print(f"\n  -- xcorr_time({label}) --")
    print(f"  pairs ({len(xc.pairs)}개): {xc.pairs}")
    print(f"  pair_at(0) shape = {xc.pair_at(0).shape}, source={xc.attrs['source']!r}")
    _report(out)

  # FFT xcorr — FFTData 두 종 각각.
  for label, src in (("raw", fft_raw_path), ("scaled", fft_scaled_path)):
    fd = FFTData.load(src)
    xc = XcorrFFTData.from_fft(fd)
    out = OUT_DIR / f"xcorr_fft_{label}.h5"
    xc.save(out)
    print(f"\n  -- xcorr_fft({label}) --")
    # v1-v2 mean peak 위치 sanity check
    vv = xc.pair("v1-v2").mean(axis=0)
    pk = int(np.argmax(vv))
    print(f"  v1-v2 peak lag={xc.lags[pk]} bin (≈ {xc.lag_hz()[pk]:.2f} Hz), ρ={vv[pk]:.3f}")
    _report(out)


def main() -> None:
  src_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Motor1000_NoLoad")
  if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
  OUT_DIR.mkdir(parents=True)

  raw_path, scaled_path = checkpoint1(src_dir)
  fft_raw_path, fft_scaled_path = checkpoint2(raw_path, scaled_path)
  checkpoint3(raw_path, scaled_path, fft_raw_path, fft_scaled_path)

  print("\n완료. 생성된 파일:")
  for p in sorted(OUT_DIR.glob("*.h5")):
    print(f"  {p}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
  main()
