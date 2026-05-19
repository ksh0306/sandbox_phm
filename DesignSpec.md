# MotorSignal 라이브러리 설계 명세서 (v2 — 클래스 계층 모델)

> 모터 3상 전압·전류(fast_adc) 신호 전처리/분석 라이브러리.
> 이 문서는 Claude Code가 그대로 구현할 수 있는 인터페이스 계약서다.
> **v1(함수형 파이프라인) 명세를 폐기하고 단계별 클래스 계층으로 재설계한다.**

---

## 0. v1 → v2 무엇이 바뀌었나 (Claude Code 필독)

| 항목 | v1 (폐기) | v2 (본 문서) |
|---|---|---|
| 처리 단계 | 모듈 레벨 순수 함수 | **각 단계가 클래스** |
| 상태/변환 | 분리 (불변 + 외부 함수) | 단계 클래스가 변환·생성 책임 보유 |
| 일반 함수 범위 | 모든 변환 | **"여러 데이터→하나로 합치기"에만 한정** |
| 데이터 dtype | float64로 캐스팅 | **fast_adc 원본 부호없는 정수 보존** |
| 축 처리 | flatten 기본화 시도 | 수신 `[패킷,채널,샘플]` 형식 유지 |
| 상속 | 없음 | `LogNormalized` IS-A 데이터 클래스 |

작업은 재작성이 아니라 **구조 전환**이다: 기존 수식(정규화/FFT/상관)은
보존하되, 함수 본문에서 단계 클래스의 생성 로직으로 옮긴다.

전 코드는 **ruff-lint를 통과**해야 한다(라인 길이/임포트 정렬/네이밍 등
프로젝트 ruff 설정 기준).

---

## 1. 클래스 계층

```
SignalData (베이스)                # 데이터 확인·시각화·저장/읽기 공통
   │
   ├── FastAdcData                 # 원본 fast_adc [패킷, 채널, 패킷당데이터], uint
   │      │
   │      └── LogNormalized        # IS-A: 정규화도 데이터의 일종 (저장/읽기 재사용)
   │
   ├── FFTData                     # 주파수 성분 [리스트, 채널, 데이터]
   │
   ├── CrossCorrLog                # Log 정규화 데이터 간 상관
   │
   └── CrossCorrFFT                # FFT 데이터 간 상관
```

### 1.1 `SignalData` (베이스 클래스)

모든 단계가 공유하는 책임만 정의. 추상 베이스.

```python
class SignalData:
    """모든 처리 단계의 공통 인터페이스: 확인 / 시각화 / 저장 / 읽기."""

    # ── 데이터 확인 ──
    def summary(self) -> dict:
        """shape, dtype, 채널명, 패킷 수 등 메타 요약 반환."""
    def describe(self) -> None:
        """사람이 읽을 수 있는 요약을 stdout 출력."""

    # ── 시각화 ──
    def plot(self, *, channels=None, ax=None, **kw):
        """단계에 맞는 기본 플롯. 하위 클래스가 구체화."""

    # ── 저장 / 읽기 (h5 우선) ──
    def to_h5(self, path: str | Path) -> None:
        """현재 인스턴스를 h5로 저장. 파일명 지정 가능."""
    @classmethod
    def from_h5(cls, path: str | Path) -> "SignalData":
        """h5에서 모든 데이터를 읽어 인스턴스 생성."""

    # ── 확장점 (v2 구현 범위 밖, 인터페이스만 고정) ──
    @classmethod
    def from_stream(cls, stream, *, field_map: dict) -> "SignalData":
        """필드명을 사전 설정한 스트림/이벤트 수신. v2에서는
        NotImplementedError로 두되 시그니처는 계약으로 고정."""
```

저장 레이아웃은 단계마다 다르므로 `to_h5`/`from_h5`는 하위 클래스에서
오버라이드한다. 단, 외부에서 보는 호출 시그니처는 동일해야 한다.

---

## 2. `FastAdcData` — 원본 데이터 (저장-읽어오기)

```python
class FastAdcData(SignalData):
    """ADC 모듈로 들어온 부호 없는 정수 fast_adc 데이터.

    data:  np.ndarray, shape = (n_packets, n_channels, n_per_packet)
           dtype = 부호 없는 정수 (uint16 등). 원본 보존, 캐스팅 금지.
    channel_names: ("v1","v2","v3","i1","i2","i3")  # 순서 고정
    bits:  int                  # ADC 비트폭 (정규화에서 사용, 지정 가능)
    fs:    float | None         # 알면 기록, 몰라도 됨
    source: str | None
    """

    def __init__(self, data, channel_names, bits, *, fs=None, source=None): ...

    @classmethod
    def from_h5(cls, path) -> "FastAdcData":
        """Motor** 폴더의 h5 포맷의 모든 데이터를 읽어온다.

        [선행 작업] 구현 전, Claude Code는 Motor** 폴더의 실제 h5
        파일을 열어 다음을 파악하고 본 문서 §2.2에 역으로 기록한다:
        - fast_adc 데이터셋의 키 경로 / shape / dtype
        - bits·채널명·fs 등 메타데이터의 attribute 키 이름
        - 패킷/채널/패킷당데이터 축 순서가 [패킷,채널,샘플] 가정과 일치하는지
        파악된 실제 규약이 본 명세의 가정과 다르면, 명세를 수정하고
        사유를 §2.2에 남긴 뒤 그 규약대로 from_h5를 구현한다."""

    def to_h5(self, path) -> None:
        """파일명을 지정하여 저장. 레이아웃 [패킷, 채널, 패킷당데이터]."""
```

**계약:**
- `data.ndim == 3`, `data.shape[1] == len(channel_names)`. 위반 시 `ValueError`.
- dtype가 부호 없는 정수가 아니면 `ValueError` (원본 보존 원칙).
- 채널 순서는 `(v1,v2,v3,i1,i2,i3)`. 스트림으로 받아도 동일 형식 전제.

### 2.2 실제 h5 구조 (Claude Code가 파일 확인 후 기록)

> Motor1000_NoLoad / Motor1000_fingerLoad / Motor7000_NoLoad 의 실제
> h5 파일(`motor_1_*_N.h5`)을 열어 아래 표를 채웠다. 이것이
> `from_h5`/`to_h5` 구현의 단일 기준이다.

| 항목 | 실제 값 (파일 확인 후 기입) |
|---|---|
| fast_adc 데이터셋 키 경로 | `/fast_adc` (루트 레벨 데이터셋) |
| 데이터셋 shape | `(n_packets, 6, 50)` — n_packets는 파일마다 ~4000±, 채널 6, 패킷당 50샘플 |
| 데이터셋 dtype | `int16` (**부호 있는** 정수) |
| bits attribute 키 | 루트 attr `adc_effective_bits` (=16). 데이터셋엔 없음 |
| 채널명 attribute 키 / 형식 | 데이터셋 attr `channel_order` = `'va,vb,vc,ia,ib,ic'` (쉼표구분 문자열) |
| fs attribute 키 (있으면) | 루트 attr `fs_hz` (=20000) |
| 축 순서 [패킷,채널,샘플] 일치 여부 | **일치** — (패킷, 채널6, 샘플50) |
| 명세 가정과의 차이 및 처리 | 아래 참조 |

**명세 가정과의 차이 및 처리:**

1. **dtype가 int16(부호 있음)** — 명세 §2는 부호 없는 정수를 가정·요구한다.
   그러나 루트 attr `adc_mid_rail=32768`, `adc_effective_bits=16`이고
   데이터가 int16 전 범위(-32768~32767)를 사용하는 점으로 보아, 실제 ADC
   코드는 **0~65535의 16bit 부호 없는 값**이며 int16은 같은 비트패턴을
   담는 저장 컨테이너일 뿐이다. 따라서 `from_h5`는 `ndarray.view(np.uint16)`
   로 **비트패턴을 그대로 유지한 채** uint16으로 재해석한다. 이는 값을
   바꾸는 캐스팅이 아니므로 "원본 보존" 원칙과 충돌하지 않으며, 동시에
   §2의 "부호 없는 정수" 계약을 만족한다. 명세 §2/§3 본문은 수정하지 않는다.
2. **채널명** — 명세 예시는 `(v1,v2,v3,i1,i2,i3)`이나 실제 파일은
   `(va,vb,vc,ia,ib,ic)`. 순서·개수(전압3+전류3)는 동일하므로 계약 위반이
   아니다. `from_h5`는 파일 실제값을 사용한다.
3. **파생 파일 제외** — `Motor7000_NoLoad/motor_1_20260511T064947.h5`
   (180MB)는 `fast_adc`가 없는 v1 파생 산출물(raw/scaled/fft/xcorr)이다.
   `FastAdcData.from_h5`는 `fast_adc`가 있는 원본 파일만 대상으로 하며,
   없으면 `ValueError`.

### 2.3 일반 함수: 다중 데이터 결합

상속이 아니라 **모듈 레벨 일반 함수**. v2에서 일반 함수는 이 용도뿐.

```python
def concat_fast_adc(items: list[FastAdcData]) -> FastAdcData:
    """동일 포맷의 여러 FastAdcData를 패킷 축(axis=0)으로 연결해
    하나의 FastAdcData를 생성. channel_names/bits/dtype 불일치 시
    ValueError. (지난주의 '여러 파일 이어붙이기'를 검증된 단계로 분리)
    """
```

---

## 3. `LogNormalized` — Log 정규화 (FastAdcData 상속, IS-A)

```python
class LogNormalized(FastAdcData):
    """fast_adc 데이터를 Log16(X+1)로 정규화한 결과.

    FastAdcData를 IS-A 상속하여 저장/읽기/시각화 인터페이스를
    그대로 재사용한다. 단 data dtype은 부동소수점.
    """

    def __init__(self, source_data: FastAdcData): ...
        # 입력 FastAdcData를 받아 정규화 수행 후 인스턴스 생성.
        # 원본은 변경하지 않는다.
```

### 3.1 정규화 수식과 출력 범위 (확정)

- 변환: `y = log16(X + 1)` , 입력 `X`는 부호 없는 정수.
- **출력 범위는 수식 그대로 `0 ~ bits/4`** (스케일 항 없음).
  16bit → 0~4, 12bit → 0~3. `bits`는 `FastAdcData.bits`에서 가져온다.
- 결과 dtype: 부동소수점(float64).

### 3.2 구현 요구 (np.log 한 줄 금지)

`np.log(X + 1) / np.log(16)`로 단순 구현하지 말 것. 두 문제를 분리 처리:

**(a) X+1 오버플로 방지**
`X`가 부호 없는 정수이므로 최댓값에서 `+1`이 0으로 래핑될 수 있다
(uint16: 65535+1 → 0). 연산 전 충분히 넓은 정수/부동소수점 타입으로
승격하거나, `log16(X+1)`를 오버플로 없는 순서로 재배열해 계산한다.

**(b) Log16 부동소수점 연산 최소화**
`log_16(n) = log_2(n) / 4` 관계를 이용한다. 정수 `X+1`의 `log_2`
정수부는 비트 길이(예: `int.bit_length` / `numpy` 동등 비트연산)로
부동소수점 없이 구하고, 가수부 보정에만 최소한의 부동소수점 `log`를
쓴다. 즉 `정수부(비트연산) + 가수부보정(소수 1회)` 형태.

### 3.3 검증 (테스트 필수)

- 비트폭별 경계: `bits=12`에서 입력 4095 → 출력 ≈ 3.0,
  `bits=16`에서 입력 65535 → 출력 ≈ 4.0 (수치 오차 허용).
- 입력 0 → 출력 0 (log16(1)=0).
- 오버플로: uint16 최댓값 배열을 넣어도 결과가 음수/0으로
  깨지지 않고 ≈ bits/4 근방.
- (b) 구현이 `np.log(X+1)/np.log(16)` 기준값과 허용 오차 내 일치.

---

## 4. `FFTData` — 주파수 성분 분석

```python
class FFTData(SignalData):
    """패킷 묶음 단위 주파수 성분.

    저장 레이아웃: [리스트, 채널, 데이터]  (채널을 필드로 보유)
    """

    def __init__(
        self,
        source: LogNormalized | list[LogNormalized],
        *,
        packets_per_group: int = 0,   # 0 = 데이터 전체를 1그룹으로
    ): ...
```

**계약:**
- 입력은 **정규화된 데이터만** (`LogNormalized` 또는 그 list).
  `FastAdcData`(미정규화)를 받으면 `ValueError`. (사용자 확정: 명세대로)
- `packets_per_group == 0`이면 전체 데이터를 한 묶음으로 FFT.
  `> 0`이면 그 패킷 개수 단위로 그룹을 나눠 그룹별 스펙트럼 산출.
- list 입력 시 각 원소를 순서대로 처리해 `[리스트, 채널, 데이터]`로 누적.
- `to_h5`/`from_h5`는 채널을 필드로 갖는 레이아웃으로 저장/복원.

**저장 레이아웃 (사용자 확정):** 채널(`va,vb,vc,ia,ib,ic`)을 **개별 필드**로
보유한다 — 클래스에서는 `channels` 딕셔너리(채널명 키, 각 값 shape
`[그룹, 데이터]`), h5에서는 `/spectrum` 그룹 아래 채널별 데이터셋으로
저장한다. `data` 프로퍼티로 `[그룹, 채널, 데이터]` 3D 배열도 얻을 수 있다.
모든 그룹은 동일 FFT 길이로 0-패딩되어 주파수 축이 일관된다.

---

## 5. `CrossCorrLog` / `CrossCorrFFT` — 상관 분석

```python
class CrossCorrLog(SignalData):
    """Log 정규화 데이터 간 채널쌍 상관 (시간영역)."""
    def __init__(
        self,
        source: LogNormalized | list[LogNormalized],
        *,
        pairs: list[tuple[str, str]] | None = None,  # None = 전체 조합
        max_lag: int | None = None,
        normalize: bool = True,
    ): ...

class CrossCorrFFT(SignalData):
    """FFT 데이터 간 채널쌍 상관 (주파수영역)."""
    def __init__(
        self,
        source: FFTData | list[FFTData],
        *,
        pairs: list[tuple[str, str]] | None = None,
    ): ...
```

두 클래스 모두 `SignalData`의 확인/시각화/저장-읽기 인터페이스를 구현한다.

**저장 레이아웃 (사용자 확정):** 채널쌍을 **개별 필드**로 보유한다 —
클래스에서는 `pair_data` 딕셔너리(`"a-b"` 키, 각 값 shape `[..., lag]`),
h5에서는 `/xcorr` 그룹 아래 쌍별 데이터셋(`va-vb` 등)으로 저장한다.
`data` 프로퍼티로 `[..., 채널쌍, lag]` 3D 배열도 얻을 수 있다.
`example.py` 파이프라인의 기본 채널쌍은 `v1-v2, v2-v3, v3-v1,
i1-i2, i2-i3, i3-i1, v1-i1, v2-i2, v3-i3` 9쌍이다.

---

## 6. 파이프라인 (확정)

```
FastAdcData ─→ LogNormalized ─→ CrossCorrLog
FastAdcData ─→ LogNormalized ─→ FFTData ─→ CrossCorrFFT
```

각 단계의 인스턴스는 독립적으로 다음이 가능해야 한다:
- 데이터 확인: `summary()` / `describe()`
- 시각화: `plot()`
- 저장/읽기: `to_h5()` / `from_h5()`

즉 파이프라인 중간 어디서든 멈춰 저장하고, 나중에 그 단계부터 다시
읽어 이어갈 수 있어야 한다.

---

## 7. 패키지 구조

```
example.py             # 실측 Motor** 폴더 일괄 처리 스크립트 (§7.1)
motorsig/
├── __init__.py        # 공개 API 재노출
├── __main__.py        # `python -m motorsig` 진입점
├── base.py            # SignalData
├── fastadc.py         # FastAdcData, concat_fast_adc
├── lognorm.py         # LogNormalized (+ Log16 구현)
├── fft.py             # FFTData
├── xcorr.py           # CrossCorrLog, CrossCorrFFT
├── cli.py             # 얇은 래퍼 (조합만, 수치 로직 금지)
└── tests/
    ├── test_fastadc.py
    ├── test_lognorm.py
    ├── test_fft.py
    └── test_xcorr.py
```

### 7.1 `example.py` — 실측 데이터 일괄 처리 (사용자 확정)

각 `Motor**` 폴더의 원본 `fast_adc` h5를 결합하고 전체 파이프라인을
적용해 폴더마다 결과 h5 5개를 생성한다. 파라미터는 스크립트 상단 상수다.

| 출력 파일 | 단계 | 비고 |
|---|---|---|
| `<folder>_raw.h5` | FastAdcData | 폴더 내 raw 파일 전체를 패킷 축으로 결합 |
| `<folder>_lognorm.h5` | LogNormalized | Log16(X+1) 정규화 |
| `<folder>_fft.h5` | FFTData | 채널별 개별 필드 |
| `<folder>_xcorr_log.h5` | CrossCorrLog | 채널쌍 9개 개별 필드 |
| `<folder>_xcorr_fft.h5` | CrossCorrFFT | 채널쌍 9개 개별 필드 |

확정 파라미터:
- **FFT 그룹 크기** `FFT_PACKETS_PER_GROUP = 800` 패킷
  (800패킷 × 50샘플 = 40000샘플 = 2초 @ 20kHz).
- **xcorr 최대 lag** `XCORR_MAX_LAG_PACKETS = 800` 패킷(2초). 실제
  `max_lag`는 패킷당 샘플 수를 곱한 샘플 단위 값으로 환산해 전달한다.
- **채널쌍 9개** (인덱스 기준, 실제 채널명 `va..ic`로 해석):
  `v1-v2, v2-v3, v3-v1, i1-i2, i2-i3, i3-i1, v1-i1, v2-i2, v3-i3`.
- 폴더명으로 시작하는 파일(= 이 스크립트 산출물)은 입력에서 제외하고,
  `fast_adc`가 없는 파생 파일은 `ValueError`로 건너뛴다.

---

## 8. 테스트 방향 (2순위 작업)

**test_fastadc.py**
- 2D/4D 입력, 채널 수 불일치 → `ValueError`
- 부호 있는/부동소수점 입력 → `ValueError` (원본 보존 원칙)
- `concat_fast_adc`: 정상 결합 shape 검증 / 메타 불일치 시 `ValueError`
- h5 라운드트립: `to_h5` → `from_h5` 결과가 원본과 bit-exact 일치

**test_lognorm.py**
- §3.3 전 항목
- IS-A 검증: `isinstance(LogNormalized(...), FastAdcData)` True
- 원본 불변: 정규화 후 입력 `FastAdcData.data` 변화 없음
- h5 라운드트립 (부동소수점 허용 오차)

**test_fft.py**
- 미정규화(`FastAdcData`) 입력 → `ValueError`
- `packets_per_group=0` → 그룹 1개 / `>0` → 그룹 수 = ceil(패킷/그룹크기)
- 단일 정현파 → 해당 주파수 빈에 피크
- 저장 레이아웃 `[리스트, 채널, 데이터]` 확인, h5 라운드트립
- 채널 개별 필드: `channels` 딕셔너리에 6채널 보유, h5는
  `/spectrum` 그룹 아래 채널별 데이터셋으로 저장

**test_xcorr.py**
- k 샘플 시프트된 채널쌍 → 시간영역 argmax lag == k
- 입력 타입 가드 (CrossCorrLog는 LogNormalized, CrossCorrFFT는 FFTData)
- 채널쌍 개별 필드: `pair_data` 딕셔너리(`"a-b"` 키) 보유, h5는
  `/xcorr` 그룹 아래 쌍별 데이터셋으로 저장

테스트는 합성 신호로 자족적으로. 실측 h5 파일 의존 금지(CI 독립 실행).

---

## 9. 작업 순서 (사용자 확정 우선순위)

0. **실제 h5 구조 파악** — Motor** 폴더의 실제 파일을 열어 §2.2 표를
   채운다. 명세 가정과 다르면 명세를 먼저 수정·기록. (다른 모든
   구현보다 선행)
1. **설계/아키텍처 확정** ← 이 문서. 클래스 계층·시그니처 동결.
2. **테스트·검증 코드 작성** — §8을 stub/xfail로 먼저 작성해 계약 고정.
3. **기존 분석 코드 이식** — 지난주 정규화/FFT/상관 수식을 각 단계
   클래스 본문으로 이식(재작성 아님). 축 형식·dtype 규약만 v2에 맞춤.
4. **팀 리뷰용 문서/PR 정리** — 이 명세서 + 변경 요약 + 사용 예제.

---

## 10. Claude Code 인계 메모

- 이 문서의 클래스 계층과 시그니처는 **계약**이다. 구현 편의로 바꾸지 말 것.
  변경이 필요하면 먼저 이 문서를 고치고 사유를 남길 것.
- **가장 먼저 §9-0을 수행**: Motor** 폴더의 실제 h5 파일을 열어 §2.2
  표를 채운 뒤에 다른 구현을 시작할 것. h5 키/축/메타 규약은 추측하지
  말고 실제 파일을 기준으로 한다.
- v1 함수형 코드가 이미 생성돼 있다면, 폐기가 아니라 **구조 전환**:
  함수 본문의 검증된 수치 로직을 해당 단계 클래스로 옮긴다.
- `LogNormalized`는 `FastAdcData`를 IS-A 상속 — 저장/읽기/시각화를
  재사용하되 dtype만 부동소수점.
- `Log16(X+1)`는 §3.2 (a)(b)를 반드시 구현. `np.log` 한 줄 금지.
  비트폭은 `FastAdcData.bits` 파라미터에서 받아 출력 0~(bits/4).
- 일반 함수는 `concat_fast_adc` 하나뿐. 그 외 변환은 단계 클래스 책임.
- 외부 의존성: numpy / h5py / (시각화) matplotlib / (테스트) pytest.
- 전 코드 ruff-lint 통과 필수.