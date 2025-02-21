from datetime import datetime

from sqlalchemy.orm import mapped_column, Mapped

from app.config.db_session import Base

# 필요시 import
from decimal import Decimal    
from sqlalchemy import ARRAY, String
from sqlalchemy.dialects.postgresql import TIMESTAMP


class SampleModel(Base): # 수정
    __tablename__ = "sample_model" # 수정
    __table_args__ = {'schema': 'gateway'}

    field1: Mapped[type] = mapped_column(primary_key=True) # 필드명 및 타입 변경, Primary Key 작성
    field2: Mapped[type] = mapped_column() # 필드명 및 타입 변경
    ...

## 필드 종류 예시
    1) 타입별 정의방법
    field: Mapped[str] = mapped_column()                               # string
    field: Mapped[float] = mapped_column(Float)                        # float; 실수형에 REAL, FLOAT, DOUBLE, NUMERIC 등 여러 타입이 존재하여 별도로 명시
    field: Mapped[int] = mapped_column()                               # int
    field: Mapped[List[str]] = mapped_column(ARRAY(String))            # List[str]
    field: Mapped[Decimal] = mapped_column()                           # 십진수; 위도/경도
    field: Mapped[datetime] = mapped_column(TIMESTAMP)                 # datetime
    
    2) 추가 옵션
    field: Mapped[type] = mapped_column(nullable=False)                # null값 비허용
    field: Mapped[type] = mapped_column(default=0)                     # 디폴트 값 정의
    field: Mapped[type] = mapped_column(index=True)                    # 인덱스 처리; 검색을 자주하는 경우 모든 열을 확인하지 않아도 되도록 인덱스 부여 가능 
    
    3) Primary Key
    longitude: Mapped[Decimal] = mapped_column(primary_key=True)       # 디폴트로 index=True, nullable=False

