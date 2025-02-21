from datetime import datetime
from sqlalchemy import String, Numeric, Float, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import mapped_column, Mapped
from app.config.db_session import Base


class ShipRouteSafetyReport(Base):
  __tablename__ = "ship_route_safety_report"
  __table_args__ = {'schema': 'gateway'}

  interest_ship_id: Mapped[str] = mapped_column(
      String(64), ForeignKey("ship_prediction_route.ship_id"), primary_key=True
  )
  report_time: Mapped[datetime] = mapped_column(primary_key=True)

  begin_time: Mapped[datetime] = mapped_column(nullable=False)
  end_time: Mapped[datetime] = mapped_column()
  start_time: Mapped[datetime] = mapped_column()

  start_longitude: Mapped[Decimal] = mapped_column()
  start_latitude: Mapped[Decimal] = mapped_column()
  arrival_time: Mapped[datetime] = mapped_column()
  arrival_longitude: Mapped[Decimal] = mapped_column()
  arrival_latitude: Mapped[Decimal] = mapped_column()

  destination_count: Mapped[int] = mapped_column(Integer)
  location_count: Mapped[int] = mapped_column(Integer)

  average_velocity: Mapped[float] = mapped_column(Float)
  upper_10_velocity: Mapped[float] = mapped_column(Float)
  lower_10_velocity: Mapped[float] = mapped_column(Float)

  unlawfulness_possibility: Mapped[float] = mapped_column(Numeric(5, 2))

  request_time: Mapped[datetime] = mapped_column(
      ForeignKey("ship_prediction_route.request_time")
  )
  prediction_route_type: Mapped[str] = mapped_column(
      String(4), ForeignKey("ship_prediction_route.prediction_route_type")
  )
