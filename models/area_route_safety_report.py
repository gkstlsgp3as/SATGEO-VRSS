from datetime import datetime
from sqlalchemy import String, Numeric, Float, Integer
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import mapped_column, Mapped
from app.config.db_session import Base


class AreaRouteSafetyReport(Base):
  __tablename__ = "area_route_safety_report"
  __table_args__ = {'schema': 'gateway'}

  interest_area_id: Mapped[str] = mapped_column(String(64), primary_key=True)
  report_time: Mapped[datetime] = mapped_column(primary_key=True)
  begin_time: Mapped[datetime] = mapped_column(nullable=False)

  end_time: Mapped[datetime] = mapped_column()
  voyage_risk_map_id: Mapped[str] = mapped_column(String(32))

  cargo_ship_identification: Mapped[int] = mapped_column(Integer)
  oil_tanker_identification: Mapped[int] = mapped_column(Integer)
  fisherboat_identification: Mapped[int] = mapped_column(Integer)
  passanger_ship_identification: Mapped[int] = mapped_column(Integer)
  etc_ship_identification: Mapped[int] = mapped_column(Integer)
  ship_identification: Mapped[int] = mapped_column(Integer)
  ship_unidentification: Mapped[int] = mapped_column(Integer)

  ship_count: Mapped[int] = mapped_column(Integer)
  destination_count: Mapped[int] = mapped_column(Integer)
  location_count: Mapped[int] = mapped_column(Integer)
  class_a_ship_count: Mapped[int] = mapped_column(Integer)
  class_b_ship_count: Mapped[int] = mapped_column(Integer)

  average_velocity: Mapped[float] = mapped_column(Float)
  upper_10_velocity: Mapped[float] = mapped_column(Float)
  lower_10_velocity: Mapped[float] = mapped_column(Float)
