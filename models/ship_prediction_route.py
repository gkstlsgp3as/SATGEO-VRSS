from sqlalchemy import String, Numeric, JSON, TIMESTAMP
from geoalchemy2 import Geometry
from sqlalchemy.orm import mapped_column, Mapped
from app.config.db_session import Base


class ShipPredictionRoute(Base):
  __tablename__ = "ship_prediction_route"
  __table_args__ = {'schema': 'gateway'}

  ship_id: Mapped[str] = mapped_column(primary_key=True)
  request_time: Mapped[str] = mapped_column(primary_key=True)
  prediction_route_type: Mapped[str] = mapped_column(primary_key=True)
  standard_prediction_time: Mapped[datetime] = mapped_column()
  start_longitude: Mapped[float] = mapped_column(Float, nullable=False)
  start_latitude: Mapped[float] = mapped_column(Float, nullable=False)
  arrival_longitude: Mapped[float] = mapped_column(Float)
  arrival_latitude: Mapped[float] = mapped_column(Float)
  rp_type: Mapped[str] = mapped_column()
  rp_requirement_second: Mapped[float] = mapped_column(Float)
  route_distance: Mapped[float] = mapped_column(Float)
  route_requirement_second: Mapped[float] = mapped_column(Float)
  route: Mapped[dict] = mapped_column(JSON, nullable=False)
  route_geom: Mapped[str] = mapped_column(Geometry("GEOMETRY"))
