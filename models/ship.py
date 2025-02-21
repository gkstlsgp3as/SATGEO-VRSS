from datetime import datetime
from sqlalchemy import String, Integer, Float, Boolean, LargeBinary
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import mapped_column, Mapped
from app.config.db_session import Base


class Ship(Base):
  __tablename__ = "ship"
  __table_args__ = {'schema': 'gateway'}

  ship_id: Mapped[str] = mapped_column(String(64), primary_key=True)
  ship_source_type: Mapped[str] = mapped_column(String(4))
  imo_no: Mapped[str] = mapped_column(String(10))
  mmsi: Mapped[str] = mapped_column(String(9))
  vpass_id: Mapped[str] = mapped_column(String(32))
  fisherboat_no: Mapped[str] = mapped_column(String(32))
  vts_id: Mapped[str] = mapped_column(String(32))
  sv_ship_id: Mapped[str] = mapped_column(String(64))
  sv_process_status: Mapped[str] = mapped_column(String(4))
  call_sign: Mapped[str] = mapped_column(String(32))
  ship_nm: Mapped[str] = mapped_column(String(100))
  ship_alias_nm: Mapped[str] = mapped_column(String(36))
  ship_type: Mapped[str] = mapped_column(String(36))
  ship_type_size: Mapped[str] = mapped_column(String(36))
  ship_type_cargo: Mapped[str] = mapped_column(String(50))
  ship_image: Mapped[bytes] = mapped_column(LargeBinary)
  built: Mapped[int] = mapped_column(Integer)
  dwt: Mapped[float] = mapped_column(Float)
  hull_type: Mapped[str] = mapped_column(String(36))
  gt: Mapped[int] = mapped_column(Integer)
  destination: Mapped[str] = mapped_column(String(50))
  eta: Mapped[datetime] = mapped_column(TIMESTAMP)
  ship_status_type: Mapped[str] = mapped_column(String(36))
  built_by: Mapped[str] = mapped_column(String(200))
  built_at: Mapped[str] = mapped_column(String(50))
  loa: Mapped[float] = mapped_column(Float)
  depth_m: Mapped[float] = mapped_column(Float)
  max_draught_m: Mapped[float] = mapped_column(Float)
  engine_built_by: Mapped[str] = mapped_column(String(200))
  designed_by: Mapped[str] = mapped_column(String(200))
  service_speed: Mapped[float] = mapped_column(Float)
  updt_dt: Mapped[datetime] = mapped_column(TIMESTAMP)
  regist_dt: Mapped[datetime] = mapped_column(TIMESTAMP)
  register_id: Mapped[str] = mapped_column(String(36))
  use_yn: Mapped[bool] = mapped_column(Boolean)
  valid_ship_code: Mapped[str] = mapped_column(String(36))
  beneficial_owner: Mapped[str] = mapped_column(String(200))
  commercial_operator: Mapped[str] = mapped_column(String(200))
  registered_owner: Mapped[str] = mapped_column(String(200))
  technical_manager: Mapped[str] = mapped_column(String(200))
  third_party_operator: Mapped[str] = mapped_column(String(200))
  nominal_owner: Mapped[str] = mapped_column(String(200))
  ism_manager: Mapped[str] = mapped_column(String(200))
  breadth: Mapped[float] = mapped_column(Float)
  teu_capacity: Mapped[int] = mapped_column(Integer)
  liquid_capacity: Mapped[int] = mapped_column(Integer)
