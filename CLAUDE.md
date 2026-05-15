# sandbox_phm — 3상 모터 PHM 데이터 분석 파이프라인

3상 유도 모터의 전압/전류 파형(HDF5) 을 받아 **전처리(bundle + cross-correlation + FFT)**
하고 **시각화(시간영역 / 주파수영역)** 까지 한 줄로 흘려보내는 워크스페이스.

## 1. 데이터 레이아웃

```
Motor1000_NoLoad/        # 1000 RPM 무부하 측정
Motor1000_fingerLoad/    # 1000 RPM 손가락 부하 측정
Motor7000_NoLoad/        # 7000 RPM 무부하 측정
  motor_*_<YYYYMMDD>T<HHMMSS>_<seq>.h5   ← raw chunk (fast_adc, uint16)
  motor_*_<YYYYMMDD>.h5                   ← 합쳐진 bundle (preproc 결과)
```

- 채널 순서 : `v1 v2 v3 i1 i2 i3` (3상 전압 + 3상 전류)
- 샘플링    : `fs = 20 kHz` (`FS_HZ`)
- raw chunk 의 `fast_adc` shape : `(n_packets, 6, 50)`. preproc 가
  `transpose(1,0,2).reshape(6, -1)` 로 풀어 `(6, N)` 배열로 만든다.

## 2. 파이프라인 한눈에

```
raw motor_*T*_*.h5 (chunks)
        │  phm_data_preproc.py
        ▼
motor_*_YYYYMMDD.h5 (bundle)
   /raw                (6, N) uint16            ─ 시간영역 raw 파형
   /scaled             (6, N) float64           ─ log16+1 정규화 (≈ [0,1])
   /xcorr_raw          시간영역 9쌍 cross-corr     (실행사항 3)
   /xcorr_scaled       시간영역 9쌍 cross-corr     (실행사항 3)
   /fft_scaled         scaled 의 단측 진폭 스펙트럼 (실행사항 5)
   /xcorr_fft_scaled   FFT 결과 9쌍 cross-corr     (실행사항 5)
        │  phm_data_vis_time.py / phm_data_vis_fft.py
        ▼
pics/<Motor folder>/*.png
```

`scaled` 정규화 공식 : `log2(raw + 1) / 16` (`log16p`).

## 3. h5 그룹 스키마

### `/xcorr_raw`, `/xcorr_scaled` (실행사항 3, `phm_data_preproc.cross_correlate_pairs`)

| 항목 | 형태 / 의미 |
|---|---|
| `attrs[samples_per_window]` | 윈도우 한 개의 샘플 수 (`SAMPLES_PER_WINDOW = 200`, 10 ms @ 20 kHz). |
| `attrs[max_lag]`            | 저장한 lag 의 최대 절댓값 [샘플]. ±100 샘플 = ±5 ms. |
| `attrs[pairs]`              | `XCORR_PAIRS` (v-v 3쌍 + i-i 3쌍 + v-i 3쌍, 총 9쌍). |
| `lags`                      | `(2·max_lag+1,)` 샘플 단위 lag 축. `lag / fs` 로 초 환산. |
| `v1-v2`, … , `v3-i3`        | `(n_win, 2·max_lag+1)` Pearson 정규 cross-correlation. |

규약 : `R_xy[k] = Σ_n x[n+k] · y[n]` (scipy 와 동일). `k > 0` 피크는
"x 가 y 보다 k 샘플 지연" 의미.

### `/fft_scaled` (실행사항 5, `compute_fft_per_channel`)

| 항목 | 형태 / 의미 |
|---|---|
| `attrs[samples_per_window]` | FFT 청크 길이 (`FFT_WINDOW_SAMPLES = 100_000`, 5 s @ 20 kHz). |
| `attrs[window]`             | `b"hann"` (스펙트럼 누수 완화용). |
| `attrs[normalization]`      | `b"amplitude_single_sided"` (`|X|·2 / Σwindow`, DC·Nyquist 는 *2 제외). |
| `attrs[source]`             | `b"scaled"`. |
| `freqs`                     | `(50001,)` 주파수 축 [Hz], `np.fft.rfftfreq`. 분해능 0.2 Hz. |
| `v1`, … , `i3`              | `(n_win, 50001)` 채널별 윈도우별 단측 진폭 스펙트럼. |

정규화 식은 "같은 진폭의 정현파라면 막대 높이가 그 진폭에 가깝다" 가
성립하도록 짜여 있어 고조파 검출에 바로 쓸 수 있다.

### `/xcorr_fft_scaled` (실행사항 5, `cross_correlate_fft_pairs`)

| 항목 | 형태 / 의미 |
|---|---|
| `attrs[samples_per_window]` | FFT 청크 길이 (lag → Hz 환산용). |
| `attrs[max_lag]`            | 저장한 lag 의 최대 절댓값 [bin]. ±500 bin = ±100 Hz. |
| `attrs[lag_unit]`           | `b"bin"`. `lag_hz = lag * fs / samples_per_window`. |
| `attrs[pairs]`              | `XCORR_PAIRS` (9개 쌍). |
| `lags`                      | `(2·max_lag+1,)` bin 단위 lag 축. |
| `v1-v2`, … , `v3-i3`        | `(n_win, 2·max_lag+1)` FFT 진폭끼리의 Pearson 정규 cross-corr. |

피크 lag = "두 채널의 평균적인 주파수 차이". 정상적인 3상 신호는 v-v / i-i
모두 lag = 0 에 강한 피크가 뜬다.

## 4. 실행 흐름

### 4.1 raw chunk → bundle 만들기

```bash
uv run python phm_data_preproc.py Motor1000_NoLoad
```

`motor_*T*_*.h5` 들을 시퀀스 순서로 합쳐 `motor_*_YYYYMMDD.h5` 를 만들고
xcorr_{raw,scaled} 와 fft_scaled, xcorr_fft_scaled 까지 한 번에 채운다.

### 4.2 이미 만들어진 bundle 에 분석 그룹만 갱신

같은 명령을 같은 폴더에 다시 돌리면 raw 가 우선 검색되지만, raw 가 없고
bundle 만 있는 폴더라면 `add_xcorr_to_bundle()` 경로로 분기해
xcorr_{raw,scaled} + fft_scaled + xcorr_fft_scaled 를 갱신한다 (raw / scaled
원본은 건드리지 않는다).

### 4.3 시각화

```bash
uv run python phm_data_vis_time.py Motor1000_NoLoad/motor_1_20260511.h5
uv run python phm_data_vis_fft.py  Motor1000_NoLoad/motor_1_20260511.h5
```

기본 출력 디렉토리는 `pics/<input 파일의 부모 폴더명>/`. `--out-dir` 로
덮어쓸 수 있다. 두 스크립트 모두 인자를 생략하면 `Motor1000_NoLoad/motor_1_20260511.h5`
를 본다.

생성되는 PNG (한 bundle 당)

| 파일명 | 내용 |
|---|---|
| `<stem>_s<start>_w<window>.png`      | 6채널 raw / scaled 파형 + cubic spline 보간 (실행사항 1–6). |
| `<stem>_xcorr_raw_heatmap.png`       | 시간영역 9쌍 cross-corr 히트맵 (raw 기준, 실행사항 4). |
| `<stem>_xcorr_raw_profile.png`       | 시간영역 9쌍 시간평균 cross-corr 프로파일. |
| `<stem>_xcorr_scaled_heatmap.png`    | 위와 동일, scaled 기준. |
| `<stem>_xcorr_scaled_profile.png`    | 위와 동일, scaled 기준. |
| `<stem>_fft_bars.png`                | 6채널 FFT 진폭 스펙트럼 막대그래프 (윈도우 평균, 0–`--max-freq` Hz). |
| `<stem>_xcorr_fft_profile.png`       | FFT 9쌍 시간평균 cross-corr 프로파일 (실행사항 6). |
| `<stem>_xcorr_fft_heatmap.png`       | FFT 9쌍 cross-corr 히트맵 (실행사항 6). |

### 4.4 모든 Motor 폴더 한 번에 (실행사항 7)

```bash
for d in Motor1000_NoLoad Motor1000_fingerLoad Motor7000_NoLoad; do
  for f in $d/motor_*.h5; do
    case "$f" in *T*) ;; *)   # raw chunk(T 포함) 는 건너뛰고 bundle 만
      uv run python -c "from phm_data_preproc import add_xcorr_to_bundle; add_xcorr_to_bundle('$f')"
      uv run python phm_data_vis_time.py "$f"
      uv run python phm_data_vis_fft.py  "$f"
    ;; esac
  done
done
```

결과는 `pics/<Motor*>/...` 에 정렬되어 저장된다.

## 5. 코드 진입점 요약

- [phm_data_preproc.py](phm_data_preproc.py)
  - `bundle(files, out_dir)` : raw chunks → bundle h5 + 모든 분석 그룹.
  - `add_xcorr_to_bundle(bundle_path)` : 분석 그룹 재계산/갱신.
  - `cross_correlate_pairs(scaled, samples_per_window, max_lag)` : 시간영역 9쌍 정규 cross-corr.
  - `compute_fft_per_channel(scaled, samples_per_window, fs)` : 채널별 단측 진폭 스펙트럼.
  - `cross_correlate_fft_pairs(fft_mag, max_lag)` : FFT 스펙트럼 9쌍 정규 cross-corr.
- [phm_data_vis_time.py](phm_data_vis_time.py)
  - `plot_signals` : 6채널 raw/scaled 시간영역 plot + cubic spline 보간.
  - `plot_xcorr_heatmaps`, `plot_xcorr_profiles` : 시간영역 9쌍 cross-corr 시각화.
- [phm_data_vis_fft.py](phm_data_vis_fft.py)
  - `plot_fft_bars` : 6채널 FFT 진폭 막대그래프 (윈도우 평균).
  - `plot_xcorr_fft_profile`, `plot_xcorr_fft_heatmap` : FFT 결과 9쌍 cross-corr 시각화.

## 6. 자주 쓰는 파라미터

| 상수 (preproc) | 의미 | 기본값 |
|---|---|---|
| `FS_HZ`                | ADC 샘플링            | 20 kHz |
| `WINDOW_MS` / `SAMPLES_PER_WINDOW` | 시간영역 xcorr 윈도우 | 50 ms → 200 샘플 (※ `FS_HZ * WINDOW_MS // 5000`) |
| `MAX_LAG_SAMPLES`      | 시간영역 xcorr lag 범위 | ±100 샘플 (±5 ms) |
| `FFT_WINDOW_SAMPLES`   | FFT 청크 길이         | 100,000 (= 5 s) |
| `FFT_MAX_LAG_BINS`     | FFT-xcorr lag 범위    | ±500 bin (±100 Hz) |
| `XCORR_PAIRS`          | 분석 채널쌍 9개       | v-v(3) + i-i(3) + v-i(3) |

| 상수 (vis) | 의미 | 기본값 |
|---|---|---|
| `DEFAULT_FILE`         | 인자 생략 시 분석할 bundle | `Motor1000_NoLoad/motor_1_20260511.h5` |
| `DEFAULT_OUT_DIR`      | PNG 저장 루트         | `pics/` |
| `DEFAULT_START` / `DEFAULT_WINDOW` (time) | 시간영역 plot 시작점·길이 | `300000` 샘플 / `100` 샘플 |
| `SPLINE_OVERSAMPLE`    | spline 보간 배수      | 20 |
| `DEFAULT_MAX_FREQ_HZ` (fft) | FFT 막대그래프 최대 표시 주파수 | 500 Hz |

## 7. 의존성

`pyproject.toml` : Python ≥ 3.13. 주요 패키지 : `numpy`, `scipy`, `h5py`, `matplotlib`.
`uv run …` 으로 실행한다.
