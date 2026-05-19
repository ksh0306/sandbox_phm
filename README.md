# MotorSignal (`motorsig`)

모터 3상 전압·전류(fast_adc) 신호 전처리/분석 라이브러리. 처리 단계마다
하나의 클래스를 두는 **단계별 클래스 계층** 구조다. 설계 계약은
[DesignSpec.md](DesignSpec.md) (v2).

## 설계 원칙

- **각 처리 단계 = 클래스** — 변환·생성 책임을 단계 클래스가 보유한다.
- **공통 인터페이스** — 모든 단계는 `SignalData`를 상속해
  `summary()` / `describe()` / `plot()` / `to_h5()` / `from_h5()`를 제공한다.
  파이프라인 중간 어디서든 멈춰 저장하고 그 단계부터 다시 읽어 이어갈 수 있다.
- **원본 dtype 보존** — `fast_adc` 원본 정수를 캐스팅 없이 유지한다.
- **일반 함수는 결합 전용** — 모듈 레벨 함수는 "여러 데이터→하나" 인
  `concat_fast_adc` 하나뿐. 그 외 변환은 단계 클래스 책임.
- **CLI는 얇게** — `cli.py`는 단계 클래스를 순서대로 호출만 한다.

## 클래스 계층

```
SignalData (베이스: 확인 / 시각화 / 저장·읽기)
   ├── FastAdcData          원본 fast_adc [패킷, 채널, 샘플], 부호없는 정수
   │      └── LogNormalized IS-A: Log16(X+1) 정규화 (float64)
   ├── FFTData              주파수 성분 [그룹, 채널, 데이터]
   ├── CrossCorrLog         Log 정규화 데이터 간 채널쌍 상관 (시간영역)
   └── CrossCorrFFT         FFT 데이터 간 채널쌍 상관 (주파수영역)
```

## 패키지 구조

```
motorsig/
├── __init__.py     # 공개 API 재노출
├── base.py         # SignalData
├── fastadc.py      # FastAdcData, concat_fast_adc
├── lognorm.py      # LogNormalized (+ log16_plus1)
├── fft.py          # FFTData
├── xcorr.py        # CrossCorrLog, CrossCorrFFT
├── cli.py          # 얇은 CLI 래퍼
└── tests/          # 합성 신호 기반 단위 테스트 (실측 데이터 불요)
```

## 파이프라인

```
FastAdcData ─→ LogNormalized ─→ CrossCorrLog
FastAdcData ─→ LogNormalized ─→ FFTData ─→ CrossCorrFFT
```

## 사용 예제

```python
import glob
from motorsig import (
    FastAdcData, LogNormalized, FFTData, CrossCorrLog, CrossCorrFFT,
    concat_fast_adc,
)

# 1) Motor** 폴더의 raw h5 적재 → 패킷 축으로 결합
files = sorted(glob.glob("Motor1000_NoLoad/motor_1_*.h5"))
raw = concat_fast_adc([FastAdcData.from_h5(f) for f in files])

# 2) Log16(X+1) 정규화 (원본 raw 는 그대로, 새 인스턴스 반환)
norm = LogNormalized(raw)

# 3) 주파수영역 분석 / 채널쌍 상관
fft = FFTData(norm, packets_per_group=200)      # [그룹, 채널, 주파수]
xc_log = CrossCorrLog(norm, max_lag=100)        # 시간영역 채널쌍 상관
xc_fft = CrossCorrFFT(fft)                      # 주파수영역 채널쌍 상관

# 각 단계는 독립적으로 확인 / 시각화 / 저장이 가능
fft.describe()
fft.to_h5("fft.h5")
fft2 = FFTData.from_h5("fft.h5")                # 그 단계부터 다시 이어가기
```

## h5 입력 포맷

`FastAdcData.from_h5`는 Motor** 폴더의 원본 파일(`motor_1_*_N.h5`)을 읽는다.
실제 파일 구조와 명세 가정과의 차이 처리는 [DesignSpec.md](DesignSpec.md) §2.2
참조 — 요지는 `fast_adc`가 int16로 저장돼 있으나 실제로는 부호 없는 16bit
ADC 코드이므로 비트패턴을 유지한 채 `uint16`으로 재해석한다.

## CLI 사용

```bash
# FFT 분석
python -m motorsig Motor1000_NoLoad/motor_1_*.h5 \
    --analysis fft --packets-per-group 200 --out fft.h5

# 시간영역 채널쌍 상관
python -m motorsig Motor1000_NoLoad/motor_1_*.h5 \
    --analysis xcorr-log --max-lag 100 --out xcorr.h5
```

## 테스트

```bash
uv run pytest            # 또는: .venv/bin/python -m pytest
```

테스트는 합성 신호만 사용하므로 실측 h5 파일 없이 CI에서 독립 실행된다
(명세서 §8). 린트는 `ruff check motorsig/`로 확인한다.
