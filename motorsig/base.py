"""SignalData: 모든 처리 단계가 공유하는 베이스 클래스."""

from __future__ import annotations

from pathlib import Path


class SignalData:
    """모든 처리 단계의 공통 인터페이스: 확인 / 시각화 / 저장 / 읽기.

    추상 베이스. 단계별 클래스가 ``summary``/``plot``/``to_h5``/``from_h5``
    를 구체화한다. ``describe``는 ``summary`` 결과로 공통 구현된다.
    """

    # ── 데이터 확인 ──
    def summary(self) -> dict:
        """shape, dtype, 채널명, 패킷 수 등 메타 요약 반환."""
        raise NotImplementedError("하위 클래스가 summary를 구현한다.")

    def describe(self) -> None:
        """사람이 읽을 수 있는 요약을 stdout 출력."""
        info = self.summary()
        print(f"[{type(self).__name__}]")
        width = max((len(str(k)) for k in info), default=0)
        for key, value in info.items():
            print(f"  {str(key):<{width}} : {value}")

    # ── 시각화 ──
    def plot(self, *, channels=None, ax=None, **kw):
        """단계에 맞는 기본 플롯. 하위 클래스가 구체화."""
        raise NotImplementedError("하위 클래스가 plot을 구현한다.")

    # ── 저장 / 읽기 (h5 우선) ──
    def to_h5(self, path: str | Path) -> None:
        """현재 인스턴스를 h5로 저장. 파일명 지정 가능."""
        raise NotImplementedError("하위 클래스가 to_h5를 구현한다.")

    @classmethod
    def from_h5(cls, path: str | Path) -> SignalData:
        """h5에서 모든 데이터를 읽어 인스턴스 생성."""
        raise NotImplementedError("하위 클래스가 from_h5를 구현한다.")

    # ── 확장점 (v2 구현 범위 밖, 인터페이스만 고정) ──
    @classmethod
    def from_stream(cls, stream, *, field_map: dict) -> SignalData:
        """필드명을 사전 설정한 스트림/이벤트 수신.

        v2 구현 범위 밖. 시그니처만 계약으로 고정한다.
        """
        raise NotImplementedError("from_stream은 v2 구현 범위 밖이다.")
