# MotorSignal 시각화 설계 명세서 (Vis)

> `motorsig` v2 단계 클래스 인스턴스를 그림으로 표현하는 시각화 모듈의
> 설계·구현 기록. 본체 설계는 [DesignSpec.md](DesignSpec.md) 참조.

---

## 0. 범위와 산출물

- **시각화 모듈** [`motorsig/visualize.py`](motorsig/visualize.py) —
  v2 클래스 인스턴스를 입력받아 `matplotlib` Figure를 만드는 순수 함수 모음.
- **일괄 실행 스크립트** [`example_visualize.py`](example_visualize.py) —
  `example.py`가 만든 파이프라인 h5를 읽어 폴더마다 PNG 5장을 저장.
- **PNG 산출물** — `pics/<folder>/` 아래. `.gitignore`의 `*.png`로 커밋 제외.

v1(`Old/`)의 시각화 코드를 **표현 방식 기준으로 적극 재활용**하되, 입력을
v1의 h5 그룹 직접 읽기에서 **v2 단계 클래스 인스턴스**로 바꿨다.

---

## 1. Old(v1) 시각화 자산의 재활용 매핑

| v2 함수 (`visualize.py`) | 출처 (Old) | 재활용한 표현 |
|---|---|---|
| `plot_waveforms` | `phm_data_vis_time.py::plot_signals` | 6채널 세로 스택, 표본점 + cubic spline 보간, raw·정규화 열 비교 |
| `plot_fft_bars` | `phm_data_vis_fft.py::plot_fft_bars` | 채널별 막대그래프, 그룹(윈도우) 평균, 고조파 보조선 |
| `plot_xcorr_heatmaps` | `phm_data_vis_*.py::plot_xcorr_*heatmap` | 3×3 그리드 `imshow`, lag=0 기준선, 공통 colorbar |
| `plot_xcorr_profiles` | `phm_data_vis_*.py::plot_xcorr_*profile` | 3×3 그리드, 행-평균 1D 곡선, 절댓값 최대 피크 표시 |

바뀐 점: v1은 `h5py`로 `xcorr_raw`/`fft_scaled` 등 그룹을 직접 열었지만,
v2는 `FastAdcData.from_h5` 등 클래스 `from_h5`로 적재한 인스턴스를 받는다.

---

## 2. 시각화 모듈 `motorsig/visualize.py`

모든 plot 함수는 Figure를 반환하며, 저장은 `save_figure`로 분리한다.

### 2.1 `plot_waveforms(signals, *, labels, start, length, channels, spline, title)`

- 입력: `FastAdcData`/`LogNormalized` 하나 또는 리스트. 리스트면 신호마다 한 열.
- `(패킷,채널,샘플)`을 `transpose(1,0,2).reshape(채널,-1)`로 평탄화해 연속
  시계열로 만든 뒤 `[start, start+length]` 구간을 그린다.
- 채널마다 표본점(`o`)과 cubic spline 보간 곡선을 함께 표시(Old 방식).
- 용도: raw ADC와 log 정규화 결과를 나란히 놓고 파형을 비교.

### 2.2 `plot_fft_bars(fft, *, max_freq, channels, fundamental, title)`

- 입력: `FFTData`. 채널별 `channels[name]`(그룹×주파수)을 그룹 축으로 평균.
- 채널마다 막대그래프 한 칸. `max_freq`로 표시 상한을 자른다.
- **DC(0 Hz) 빈 제외** — 정규화 데이터의 DC 성분이 압도적으로 커
  나머지 막대를 안 보이게 만들기 때문(`freqs > 0` 마스크).
- `fundamental`을 주면 그 정수배 위치에 고조파 보조선.

### 2.3 `plot_xcorr_heatmaps(xcorr, *, freq_resolution, title)`

- 입력: `CrossCorrLog`/`CrossCorrFFT`. `pair_data`의 쌍별 `(행, lag)` 행렬을
  3×3 `imshow` 그리드로. 행 축 = 입력 항목/그룹 인덱스.
- 색상 규칙:
  - 데이터에 음수가 있으면(정규화된 상관) `RdBu_r`, 0 중심 대칭.
  - 모두 0 이상이면 `viridis` + **로그 색상 스케일**(`LogNorm`).
    `CrossCorrFFT`는 미정규화(§3 참조)라 lag 0 첨두가 압도적으로 크다.
    선형 색상이면 첨두 한 줄만 보이므로, 로그 색상으로 첨두 주변
    falloff와 그룹별 변화를 드러낸다.

### 2.4 `plot_xcorr_profiles(xcorr, *, freq_resolution, title)`

- 입력: `CrossCorrLog`/`CrossCorrFFT`. 쌍별 행렬을 행 축으로 평균한 1D
  프로파일을 3×3 그리드 선그래프로. 절댓값 최대 lag를 빨간 점 피크로 표시.

### 2.5 lag 축 환산 (`_lag_axis`)

| 입력 | lag 원단위 | 표시 단위 | 환산 |
|---|---|---|---|
| `CrossCorrLog` | 샘플 | ms | `lag / fs * 1000` |
| `CrossCorrFFT` | 주파수 빈 | Hz | `lag * freq_resolution` |

`CrossCorrFFT`는 빈→Hz 환산에 필요한 주파수 분해능을 자체 보유하지
않으므로, 호출 측이 원본 `FFTData.freqs`에서 `freqs[1]-freqs[0]`을
계산해 `freq_resolution` 인자로 전달한다(클래스 미변경 결정 — §3).

---

## 3. 설계 결정과 사유

1. **plot 텍스트는 영문** — matplotlib 기본 폰트(DejaVu Sans)에 한글
   글리프가 없어 제목/축을 한글로 쓰면 □로 깨진다. 코드 주석·docstring은
   한글, 그림 안 텍스트는 영문으로 통일.
2. **`Agg` 백엔드 강제** — 헤드리스 환경에서 PNG 저장만 하므로
   `visualize.py` 임포트 시 `matplotlib.use("Agg")`.
3. **`CrossCorrFFT` 클래스 미변경** — FFT 상관의 lag를 Hz로 표시하려면
   주파수 분해능이 필요하나, v2 §5 `CrossCorrFFT` 시그니처를 바꾸지 않기
   위해 클래스에 필드를 추가하지 않고 호출 측이 `FFTData`에서 분해능을
   넘기는 방식을 택했다(DesignSpec.md §10 계약 보존).
4. **`CrossCorrLog`는 프로파일만** — `example.py` 파이프라인의
   `CrossCorrLog`는 결합 신호 1개에 대한 상관이라 항목이 1개뿐이다.
   `(1 × lag)` 히트맵은 한 줄짜리라 의미가 없어 프로파일만 생성한다.
   `CrossCorrFFT`는 그룹이 여러 개(예: 21)라 히트맵·프로파일 모두 생성.
5. **시각화 모듈은 `__init__` 비노출** — `motorsig` 코어 임포트가
   matplotlib까지 끌어오지 않도록 `motorsig/__init__.py`에서 재노출하지
   않는다. 사용 시 `from motorsig.visualize import ...`로 명시 임포트.

---

## 4. 일괄 실행 스크립트 `example_visualize.py`

각 `Motor**` 폴더의 파이프라인 h5 5개(`_raw`/`_lognorm`/`_fft`/
`_xcorr_log`/`_xcorr_fft`)를 읽어 `pics/<folder>/`에 PNG 5장을 만든다.

| PNG | 함수 | 내용 |
|---|---|---|
| `<folder>_waveform.png` | `plot_waveforms` | raw vs log 정규화 6채널 시간영역 파형 |
| `<folder>_fft_bars.png` | `plot_fft_bars` | 채널별 진폭 스펙트럼 막대그래프 |
| `<folder>_xcorr_log_profile.png` | `plot_xcorr_profiles` | 시간영역 채널쌍 상관 프로파일 (9쌍) |
| `<folder>_xcorr_fft_heatmap.png` | `plot_xcorr_heatmaps` | 주파수영역 상관 히트맵 (그룹×lag) |
| `<folder>_xcorr_fft_profile.png` | `plot_xcorr_profiles` | 주파수영역 상관 프로파일 (9쌍) |

**전기 기본 주파수 추정** — `electrical_fundamental`은 폴더명
`Motor<rpm>_*`에서 rpm을 뽑아 `전기주파수 = rpm/60 × 극쌍수(7)`로
계산해 `plot_fft_bars`의 고조파 보조선과 표시 상한(`max_freq`)에 쓴다.
극쌍수 7은 원본 h5 루트 attr `pole_pairs`에서 확인한 값이다.

실행:

```bash
python example.py            # 먼저 파이프라인 h5 생성
python example_visualize.py  # pics/<folder>/ 아래 PNG 15장 생성
```

---

## 5. 의존성

`numpy`, `scipy`(`CubicSpline`), `matplotlib`. 모두 `pyproject.toml`에
이미 선언돼 있다. 전 코드 `ruff check` 통과.
