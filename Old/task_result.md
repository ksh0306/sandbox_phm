# task_result.md — phm.py 구현 과정과 판단 기록

`implement.md`의 목표(전압3상·전류3상 파형의 raw / log16 / FFT 비교 시각화)를 [`phm.py`](phm.py)로 구현하면서 거친 의사결정 흐름과 시행착오를 모두 정리한다. 동일 결론에 다른 사람이 스스로 도달할 수 있도록 "왜"에 무게를 둔다.

## 0. 출발점

목표(`implement.md`):
1. h5 파일에서 전압3상·전류3상 파형 추출
2. raw / log16 / log16의 FFT 세 표현을 반환하는 함수
3. 세 표현을 비교 시각화

이미 있던 자산:
- [`sandbox.py`](sandbox.py): h5 로드, 패킷 평균(50:1), 시간축 플롯의 모범 사례
- [`fftsample.py`](fftsample.py): `log16(x) = log2(x+ε)/4` 변환 + FFT 패턴 (ε=1e-9, 0 발산 방지)

## 1. h5 파일 구조 파악 (먼저 할 일)

설계를 시작하기 전에 데이터의 모양을 직접 확인한다. `h5py`로 keys/attrs/dtype을 덤프:

```python
with h5py.File("motor_0_20260429T020906_0.h5", "r") as f:
    print(dict(f.attrs))
    for k in f.keys():
        d = f[k]
        print(k, getattr(d, "shape", None), getattr(d, "dtype", None))
```

알게 된 핵심 사실:
- `fast_adc`: shape `(n_packets, 6, 50)`, dtype `int16`, channel order `va,vb,vc,ia,ib,ic`
- attrs: `fs_hz=20000`, `samples_per_pkt=50`, `pole_pairs=7`, `slow_ctx_hz=50`
- `ts_us`: shape `(n_packets,)`, uint64, "TIM2 free-running counter (us)"
- `slow_ctx`: shape `(n_packets/8 정도,)`, compound dtype에 `erpm`, `tachometer`, `s1_rx_ts_us` 등 포함
- `events`: 비어 있음 (shape (0,))

이 단계에서 **데이터셋 vs 어트리뷰트의 위치**를 확실히 구분해 두면 뒤의 혼동을 줄일 수 있다.

## 2. 1차 구현: extract + 6×3 grid 시각화

설계 결정:
- 함수가 받는 입력: 파일 경로 / 반환: dict (raw/scaled/fft_mag/fft_freq/t/fs_hz/...)
- raw는 ADC 원본 의미를 살리기 위해 `int16 → uint16` reinterpretation (sandbox.py와 동일 관습). `fftsample.py`도 0~65535 범위에서 log16을 적용하므로 그 일관성 유지.
- (n_packets, 6, 50)을 `(6, n_packets*50)`로 reshape (sandbox.py 패턴)
- 실수 신호이므로 `np.fft.rfft` 사용 → 절반 스펙트럼 + magnitude

시각화는 6채널 × 3표현(raw / log16 / FFT) 그리드. 시간축은 처음 100ms로 잘라서 표시.

이 시점의 결과물은 동작은 하지만 **"신호 특징이 안 보인다"** — PWM 노이즈에 묻혀 사인파 모양이 안 드러난다.

## 3. 2차 요구: "5사이클만 보여줘"

문제는 **사이클 개수를 정하려면 기본주파수를 알아야 한다**는 것. 처음 시도한 접근은 "FFT에서 자동 검출":

```python
# 첫 시도: log16(raw)의 평균 FFT에서 최대 피크 = 기본주파수
spec = fft_mag[3:6].mean(axis=0)   # 전류 3채널 평균
band = (fft_freq >= 5) & (fft_freq <= fs_hz/2)
peak = fft_freq[band][np.argmax(spec[band])]
```

결과: `f0 ≈ 8017, 9599, 10000 Hz` 같은 값이 나옴. **PWM 스위칭 주파수에 사로잡혔다.** 모터 전기 기본주파수는 보통 수십~수백 Hz라서 PWM 영역(수~수십 kHz)이 spec 상위에 자리잡으면 검출이 빗나간다.

### 보정 시도: 다운샘플로 PWM 제거 후 검출

`sandbox.py`가 패킷 평균(50:1) 다운샘플로 PWM을 평탄화해 사인을 드러내는 점에서 힌트를 얻어, 전류 3채널을 `decim=10`으로 평균 다운샘플 → fs_pkt=2000Hz, f_max=1000Hz로 제한해서 다시 검출:

```python
n = raw.shape[1] - (raw.shape[1] % decim)
pkt = raw[3:6, :n].astype(np.float64).reshape(3, -1, decim).mean(axis=2)
pkt = pkt - pkt.mean(axis=1, keepdims=True)  # DC 제거
fs_pkt = fs_hz / decim
```

검출값: 83 Hz, 783 Hz, 222 Hz, 116.6 Hz 등 — **모터 영역으로 들어왔다**. 그러나 검출 안정성이 파일마다 들쑥날쑥했다(같은 모터인데 어떤 파일은 116Hz, 어떤 파일은 783Hz).

## 4. 결정적 전환: 사용자 힌트 "ts_us를 써라"

여기서 사용자가 "FFT로 검출하지 말고 `ts_us`를 써라"고 알려줌. 즉시 수용하지 않고 **검증**부터 했다. ts_us 자체가 주기를 담고 있는지를 확인하는 게 핵심.

### 4-1. ts_us의 형태를 본다

```python
ts = f["ts_us"][:].astype(np.int64)
d  = np.diff(ts)
print(d.mean(), d.std(), d.min(), d.max())
v, c = np.unique(d, return_counts=True)  # 빈도 상위 값
```

관찰:
- 평균 2500 us = 1/(20000/50) → 그냥 패킷 획득 간격(40 0Hz packet rate)
- 분포는 2500 ± 50 us 범위에 quantization된 값들 + 가끔 100000us의 큰 갭(idle)

### 4-2. ts_us diff의 FFT로 모터 주기 검출이 가능한가?

```python
d = np.diff(ts).astype(np.float64) - 2500.0  # mean 제거
spec = np.abs(np.fft.rfft(d))
freq = np.fft.rfftfreq(len(d), d=1.0/400.0)  # fs_pkt=400Hz
```

상위 피크: 200 Hz(Nyquist 부근), 150 Hz, 96.96 Hz, 145 Hz...

검증을 위해 **신뢰 가능한 ground truth**가 필요. h5 안을 다시 뒤져 `slow_ctx['erpm']`을 본다 — 6999가 거의 일정. 즉 **f_elec = 6999/60 = 116.65 Hz**가 정답.

ts_us diff의 FFT 상위 피크에 116.65 Hz가 없다 → **ts_us 자체에는 모터 전기주기가 신뢰할 수 있는 신호로 들어 있지 않다.** 200 Hz 근처 피크는 packet quantization noise, 100Hz/150Hz는 PWM 별칭(aliasing) 의심.

추가로 `tachometer` 필드 검증:
- 20ms 간격으로 14~15씩 증가 → 700 pulses/s
- 700 / 116.65 ≈ 6 → BLDC 6-step commutation과 정확히 일치 → **erpm이 ground truth로 신뢰 가능함**

### 4-3. 사용자에게 데이터로 다시 묻는다

"틀렸다"고 단정하지 않고 **분석 결과(피크 위치, erpm 값, tachometer 일치)를 보여주고** 어느 소스를 쓸지 선택지를 제시. 사용자는 `slow_ctx.erpm`으로 합의.

**교훈:** 사용자의 힌트와 실제 데이터가 충돌하면 데이터를 보여주고 합의. 추측으로 진행하지 않는다.

## 5. 최종 구현

`extract_phases`에 추가/변경:

```python
# slow_ctx.erpm 중앙값으로 모터 전기주파수 산출
erpm = sc["erpm"] if len(sc) else np.array([], dtype=np.int32)
erpm_active = erpm[erpm > 0]                 # 정지 구간 제외
erpm_median = float(np.median(erpm_active)) if len(erpm_active) else 0.0
f_elec_hz = erpm_median / 60.0 if erpm_median > 0 else None
```

`detect_fundamental_hz`는 제거. `plot_compare`는 dict의 `f_elec_hz`를 기본값으로 쓰되 인자 `fundamental_hz`로 강제 가능.

### 엣지 케이스 두 가지를 잡아야 했다

**(A) 정지 구간이 많은 파일** — `motor_1_..._1.h5`, `_5.h5`는 erpm이 대부분 0이고 짧은 구간만 회전. `np.median(erpm)`은 0이 되어 버린다. → **0이 아닌 값들의 중앙값**을 쓴다.

**(B) idle gap으로 fs 평균이 망가지는 파일** — 처음에 `ts_us`로 정확한 fs 산출을 시도하면서 `mean(diff)`를 썼는데, `motor_0_..._0.h5`는 100ms 갭이 다수 포함되어 평균이 18ms로 부풀어 fs가 2688Hz로 나옴. 그러나 갭은 "샘플 사이의 idle"이지 샘플 간격이 아니다.

해결: **nominal 패킷 간격의 ±50% 범위 안에 들어오는 diff만 평균**.

```python
nominal_pkt_us = samples_per_pkt * 1e6 / fs_hz_nominal   # = 2500
mask = (diffs > 0.5*nominal_pkt_us) & (diffs < 1.5*nominal_pkt_us)
fs_hz = samples_per_pkt / (diffs[mask].mean() / 1e6)
```

이렇게 하면 거의 모든 파일에서 fs ≈ 20000.0 Hz로 안정. (실제로 20000.10/19999.93 등 ppm 단위 클럭 편차도 잡아냄.)

## 6. 최종 결과 요약

| 파일 | f_elec | window | 비고 |
|---|---|---|---|
| `motor_0_..._0` | n/a | 100 ms 폴백 | 모터 전부 정지 (erpm=0) |
| `motor_1_..._1` | 71.23 Hz | 70.2 ms | 활성 구간 erpm 중앙값 |
| `motor_1_..._2` | 116.67 Hz | 42.9 ms | |
| `motor_1_..._3` | 116.65 Hz | 42.9 ms | |
| `motor_1_..._4` | 116.65 Hz | 42.9 ms | |
| `motor_1_..._5` | 62.28 Hz | 80.3 ms | 활성 구간 erpm 중앙값 |

전류(ia/ib/ic) 행에서 5사이클 사인 파형이 깔끔히 보이는 게 핵심 검증 포인트.

## 7. 따라하기 위한 체크리스트

- [ ] h5 구조부터 보기: attrs / keys / dtype / 첫 몇 행. 데이터셋과 어트리뷰트 위치를 적어 두기.
- [ ] 도메인 사실을 먼저 확보: BLDC라면 PWM 스위칭과 전기 기본주파수의 분리, 6-step commutation 등.
- [ ] **신호처리로 도메인 값을 추정하기 전에 메타 필드(`erpm`, `tachometer` 등)를 먼저 살피기.** 신호처리는 메타가 없을 때 보조.
- [ ] 자동검출이 미덥지 않으면 ground truth와 cross-check (여기선 `tachometer` 증분 vs erpm).
- [ ] 사용자 힌트와 데이터가 충돌하면 분석 결과를 들이밀고 다시 합의. 추측 금지.
- [ ] 통계량(mean/median)을 쓰기 전에 idle/0 같은 outlier 모드부터 점검:
  - fs는 nominal 간격 근처만 평균 (idle gap 제외)
  - erpm 중앙값은 0 제외 후 계산

## 8. 확장 아이디어 (이번 구현 범위 밖)

- erpm이 시간에 따라 크게 변하면 `s1_rx_ts_us`로 표시 윈도우 시점의 erpm을 sampling
- log16 외에 cube-root 등 다른 비선형 변환 비교
- FFT를 STFT(spectrogram)로 바꿔 시간-주파수 변동 시각화
