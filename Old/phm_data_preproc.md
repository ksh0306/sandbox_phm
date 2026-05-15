# PHM 데이터 전처리 & 시각화 구현 문서

PHM(Prognostics & Health Management) 모터 진단을 위한 다채널 ADC 데이터의
전처리 파이프라인과 시각화 도구의 구현 과정을 정리한 문서.

## 1. 입력/출력 개요

### 1.1 원시 입력 (`motor_<id>_<date>_<seq>.h5`)
- 형식: HDF5
- 데이터셋: `fast_adc` — shape `(n_packets, 6, 50)`, dtype `int16`
- 채널 순서: `v1, v2, v3, i1, i2, i3` (3상 전압 + 3상 전류)
- 샘플링: 20 kHz, 패킷 주기 2.5 ms (패킷당 50샘플 = 2.5 ms × 20 kHz)
- 파일은 시퀀스(`_1`, `_2`, ...)로 분할되어 시간 순으로 연결되어야 함

### 1.2 전처리 출력 (`motor_<id>_<date>.h5`)
출력 파일명은 첫 입력 파일의 첫 3토큰(`motor_<id>_<date>`)으로 통일하며
시퀀스 번호 토큰을 떼어 단일 파일로 묶는다.

| 데이터셋          | shape                 | 설명 |
|-------------------|-----------------------|------|
| `raw`             | `(6, N)`              | 채널별 원시 ADC (uint16) |
| `scaled`          | `(6, N)`              | `raw`의 log16 압축본 (float64) |
| `raw_fft`         | `(6, n_chunks, n_bins)` | 50 ms 청크 rFFT magnitude |
| `scaled_fft`      | `(6, n_chunks, n_bins)` | log16의 청크 rFFT magnitude |
| `fft_freq`        | `(n_bins,)`           | FFT 주파수 축 [Hz] |
| `xcorr`           | `(3, n_chunks, 2W-1)` | 상별 (v_k ★ i_k) 청크 교차상관 |
| `xcorr_fft`       | `(3, n_chunks, W)`    | xcorr 청크의 rFFT magnitude |
| `xcorr_lag_s`     | `(2W-1,)`             | 교차상관 lag 축 [s] |
| `xcorr_fft_freq`  | `(W,)`                | xcorr_fft 주파수 축 [Hz] |

부수 속성(attrs): `fs_hz`, `window_ms`, `samples_per_window`, `channels`,
`xcorr_pairs`, `source_files`.

여기서 `W = SAMPLES_PER_WINDOW = 1000`(50 ms × 20 kHz), `n_bins = W/2 + 1 = 1001`.

---

## 2. 데이터 생성 — [phm_data_preproc.py](phm_data_preproc.py)

`implement.md`의 1~6단계 파이프라인을 그대로 따르되,
xcorr 단계를 추가하여 상별 v–i 위상 관계까지 함께 묶는다.

### 2.1 상수와 채널 메타
[phm_data_preproc.py:23-27](phm_data_preproc.py#L23-L27)
```python
CHANNEL_NAMES = ["v1", "v2", "v3", "i1", "i2", "i3"]
LOG16_EPSILON = 1e-9
FS_HZ = 20000
WINDOW_MS = 50
SAMPLES_PER_WINDOW = FS_HZ * WINDOW_MS // 1000  # = 1000
```
- 50 ms 윈도우는 60 Hz 기본파의 약 3주기를 포함 → 기본파/고조파를
  안정적으로 잡을 수 있는 최소 길이로 선택.
- 코드 주석에는 "100ms"가 일부 남아 있지만 실제 구현은 `WINDOW_MS = 50`이며
  출력 메타에도 `window_ms = 50`으로 기록된다.

### 2.2 채널 분리 — `load_phases` ([phm_data_preproc.py:35-42](phm_data_preproc.py#L35-L42))
```python
d = f["fast_adc"][:]            # (n_packets, 6, 50)
d.transpose(1, 0, 2).reshape(6, -1)
```
- `transpose(1, 0, 2)`로 채널 축을 가장 앞으로 → `(6, n_packets, 50)`.
- `reshape(6, -1)`로 패킷을 시간축으로 평탄화 → `(6, N)`.
- 이 순서가 원본 데이터의 시간 순서를 보존한다(패킷 i의 0~49번 샘플이
  패킷 i+1의 0~49번 샘플 앞에 와야 함).

### 2.3 파일 연결과 정렬 ([phm_data_preproc.py:97-117](phm_data_preproc.py#L97-L117))
- 정규식 `^([^_]+_[^_]+_[^_]+)_.+\.h5$`로 4토큰 형식의 입력만 인정.
- `_(\d+)\.h5$`로 끝의 시퀀스 번호를 추출해 정렬 키로 사용.
- 재실행 시 자기 자신(3토큰 출력)을 다시 입력으로 잡지 않도록 정규식 매칭으로 분리.
- 모든 파일을 `np.concatenate(axis=1)`로 시간축 연결.

### 2.4 청크 단위 FFT — `chunked_fft` ([phm_data_preproc.py:45-57](phm_data_preproc.py#L45-L57))
- 시계열을 `samples_per_window` 단위로 잘라 마지막 청크의 꼬리는 잘라낸다
  (불완전 청크 제외).
- `reshape(6, n_chunks, W)` 후 `np.fft.rfft(axis=2)`의 magnitude를 반환.
- 결과 shape: `(6, n_chunks, n_bins)`.

### 2.5 log16 압축 — `log16` ([phm_data_preproc.py:30-32](phm_data_preproc.py#L30-L32))
```python
np.log2(x + LOG16_EPSILON) / 16.0
```
- uint16 ADC 값(0 ~ 65535)을 약 [0, 1] 범위로 압축.
- `+ epsilon`은 0 입력에 대한 log 발산 방지.
- 동적 범위가 큰 신호의 작은 변동을 시각/학습 모두에서 살리기 위함.

### 2.6 상별 청크 교차상관 — `chunked_xcorr_pairs` ([phm_data_preproc.py:60-94](phm_data_preproc.py#L60-L94))

목적: 같은 상의 전압/전류 사이 시간 지연·위상 관계를 청크 단위로 추적.

DSP 표준 관례 (Oppenheim/Schafer):
```
r[m] = Σ_n v[n] · i[n+m]
```
- 양의 lag → i가 v보다 m 샘플 지연(전류가 전압보다 늦음, 인덕티브)
- 음의 lag → i가 v보다 m 샘플 앞섬

**FFT 기반 batched 구현:**
1. 각 상의 v, i 시계열을 `(n_pairs, n_chunks, W)`로 reshape.
2. 선형(비순환) 교차상관을 위해 길이 `2W`로 zero-pad 후 rFFT.
3. `IFFT[conj(V) · I]`로 모든 lag을 한 번에 계산.
4. 순환 출력에서 인덱싱:
   - 양의 lag `0..W-1` → 인덱스 `0..W-1`
   - 음의 lag `-(W-1)..-1` → 순환 wrap된 인덱스 `W+1..2W-1`
5. scipy 관례를 따라 `[-(W-1) ... -1, 0, 1 ... W-1]` 순서로 결합 → `(2W-1)`.

각 청크의 윈도우 길이가 50 ms이므로 lag 범위는 ±49.95 ms.

### 2.7 xcorr 청크의 FFT ([phm_data_preproc.py:134](phm_data_preproc.py#L134))
- 교차상관 결과 자체에 청크별 rFFT를 다시 적용 → 위상 결합도의
  주파수 분포(고조파에서 v–i 결합이 어떻게 분산되는지)를 본다.
- `n` 입력은 `2W-1`로 그대로 두며 `n_bins = W` (홀수 길이의 rFFT bin 개수).

### 2.8 메타 + 저장 ([phm_data_preproc.py:142-158](phm_data_preproc.py#L142-L158))
- 모든 큰 데이터셋에 `compression="gzip"` 적용.
- `fft_freq`/`xcorr_lag_s`/`xcorr_fft_freq`는 작은 1D 축이라 비압축.
- `xcorr_pairs`로 페어 이름(`v1*i1`, `v2*i2`, `v3*i3`)을 명시.

---

## 3. 시각화 1: 종합 — [phm_data_preproc_visualization.py](phm_data_preproc_visualization.py)

조밀한 시계열·청크 FFT·청크 xcorr를 사람이 한눈에 보기 위한 3종 그림 생성.

### 3.1 픽셀 단위 min/max envelope — `envelope_minmax` ([phm_data_preproc_visualization.py:18-36](phm_data_preproc_visualization.py#L18-L36))

문제: 수십~수백만 샘플을 일반 line plot으로 그리면 점이 겹쳐서 피크가 사라짐.

해결:
1. 시계열을 `target_width` 픽셀 기둥으로 나눔.
2. 각 기둥에서 (min, max)를 계산해 `fill_between`으로 채움.
3. → spike/peak이 한 픽셀 안에 항상 살아남는다.

이는 오디오 파형 표시기에서 표준으로 쓰는 전형적인 down-sampling 방식.

### 3.2 LogNorm 보조 — `_log_norm` ([phm_data_preproc_visualization.py:39-48](phm_data_preproc_visualization.py#L39-L48))
- FFT magnitude는 동적 범위가 매우 큼 → 선형 스케일에서는 큰 값만 보임.
- `vmin`은 양수값의 1퍼센타일로 잡아 노이즈 floor를 끊고,
  `vmax`는 양수값의 최대치로 설정.
- 0/음수는 `floor`로 클램프 처리해 LogNorm이 깨지지 않게 함.

### 3.3 종합 그림 — `plot_bundle` ([phm_data_preproc_visualization.py:51-145](phm_data_preproc_visualization.py#L51-L145))
6채널 × 4열 grid:
1. **col 1**: raw min/max envelope (시간 [s] vs ADC).
2. **col 2**: raw 청크 FFT 스펙트로그램 — `pcolormesh` + LogNorm + magma cmap.
3. **col 3**: log16 envelope.
4. **col 4**: log16 청크 FFT 스펙트로그램.

핵심 디테일:
- 청크 시간축은 `(np.arange(n_chunks) + 0.5) * window_s` — 청크 중심.
- 각 행에 채널 라벨 + Voltage/Current 구분 표시.
- 출력: `<stem>_viz.png`.

### 3.4 xcorr heatmap — `plot_xcorr` ([phm_data_preproc_visualization.py:148-225](phm_data_preproc_visualization.py#L148-L225))
3 페어 × 2 열:
- **좌**: lag-time heatmap. signed value이므로 `RdBu_r` diverging cmap에
  `±99 percentile`로 클립하여 outlier가 색상 범위를 잡아먹지 않게 함.
- **우**: xcorr_fft 스펙트로그램 (LogNorm + magma).
- y축은 lag [ms], `axhline(0)`으로 zero-lag 기준선 표시.

### 3.5 xcorr 라인 요약 — `plot_xcorr_lines` ([phm_data_preproc_visualization.py:228-320](phm_data_preproc_visualization.py#L228-L320))
heatmap은 전반적 패턴은 잘 보여주지만 정량 추적이 어려워 라인 그래프 보완.

3 페어 × 3 열:
- **col 1**: 시간평균 xcorr 프로파일 (lag vs 평균값).
  → 평균적인 v–i 위상 관계의 모양.
- **col 2**: 청크별 peak lag (좌축, 빨강) + peak |xcorr| (우축, 회색·log).
  → 시간이 흐르면서 페이크 lag/세기가 어떻게 변하는지.
- **col 3**: 시간평균 xcorr FFT 스펙트럼 (log y).
  → 위상 결합도가 어느 주파수에 집중되는지.

`np.argmax(|xcorr|, axis=2)` + `np.take_along_axis`로
청크별 peak를 vectorized 추출.

---

## 4. 시각화 2: 비트맵 정밀도 — [phm_data_preproc_vis2.py](phm_data_preproc_vis2.py)

`plot_bundle`/`plot_xcorr`가 사용하는 `pcolormesh`는 셀 경계가
픽셀과 정렬되지 않아 미세한 줄무늬가 흐려질 수 있음.
이를 보완해 셀 = 픽셀 1:1 비트맵으로 표현.

### 4.1 핵심 차이
- `imshow(..., interpolation="nearest", aspect="auto")` 사용.
- `extent=[0, duration_s, 0, f_max]`로 축 단위만 설정하고 보간은 끔.
- 결과: 청크 경계, FFT bin 경계가 픽셀 단위로 정확히 살아남아
  스펙트럼 라인의 미세 구조 파악에 유리.

### 4.2 두 가지 출력
- `plot_fft_bitmap`: 6채널 × 2열(raw_fft / scaled_fft), `<stem>_fft_bitmap.png`.
- `plot_xcorr_bitmap`: 3페어 × 2열(xcorr / xcorr_fft), `<stem>_xcorr_bitmap.png`.

---

## 5. 실행 흐름

### 5.1 전처리 실행
```
python phm_data_preproc.py
```
- 현재 디렉토리에서 `motor_*_*_<seq>.h5` (4토큰) 파일을 시퀀스 순으로 모음.
- 단일 `motor_<id>_<date>.h5` (3토큰)로 묶어 저장.

### 5.2 시각화 실행
```
python phm_data_preproc_visualization.py            # 인자 없음 → 현재 dir 자동 탐색
python phm_data_preproc_visualization.py FILE.h5    # 특정 파일만

python phm_data_preproc_vis2.py                     # 비트맵 시각화
python phm_data_preproc_vis2.py FILE.h5
```
입력 후보 판별: 파일명을 `_`로 split했을 때 토큰 수가 3인 것만 전처리 출력으로 인정.

### 5.3 산출물 (예시: `motor_1_20260506T053537.h5`에 대해)
| 파일                                      | 생성 스크립트                       |
|-------------------------------------------|-----------------------------------|
| `motor_1_20260506T053537.h5`              | phm_data_preproc.py               |
| `..._viz.png`                             | phm_data_preproc_visualization.py |
| `..._xcorr_viz.png`                       | phm_data_preproc_visualization.py |
| `..._xcorr_lines.png`                     | phm_data_preproc_visualization.py |
| `..._fft_bitmap.png`                      | phm_data_preproc_vis2.py          |
| `..._xcorr_bitmap.png`                    | phm_data_preproc_vis2.py          |

---

## 6. 설계 노트와 트레이드오프

### 6.1 윈도우 길이 50 ms
- 60 Hz 기준 3주기 — 기본파/저차 고조파 추정의 최소치 근방.
- 더 길게 잡으면 주파수 해상도 ↑ 시간 해상도 ↓.
- 모터 부하의 빠른 변동을 청크 단위로 추적하기 위해 짧은 윈도우 채택.

### 6.2 log16 압축
- ADC 동적 범위가 매우 커서 raw 그대로는 작은 변동이 묻힘.
- log 압축으로 신경망 학습 시에도 입력 분포의 꼬리가 짧아지는 이점.

### 6.3 FFT 기반 xcorr
- 시간 영역 직접 계산 대비 `O(W²) → O(W log W)`.
- 청크 batched + zero-pad로 모든 페어/청크를 한 번의 rFFT/iFFT로 처리.

### 6.4 시각화 이중화 (pcolormesh + imshow)
- pcolormesh: 비균일 시간/주파수 축에 강함, 그러나 보간으로 미세 라인 흐려짐.
- imshow(nearest): 픽셀 정확도가 필요할 때.
- 둘 다 두는 이유: 동일 데이터를 두 관점에서 검증 + 보고서용 표현 다양화.

### 6.5 하위 호환과 재실행 안전성
- 입력 파서가 4토큰만 인정하므로 출력(3토큰)이 다음 실행에서
  입력으로 다시 잡히지 않는다.
- 시각화 스크립트는 반대로 3토큰만 인정 → 두 스크립트의 입력 도메인이 분리되어
  파이프라인이 안전하게 idempotent.
