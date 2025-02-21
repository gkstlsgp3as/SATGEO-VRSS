from datetime import datetime
from sqlalchemy import String, Numeric, Float
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import mapped_column, Mapped
from app.config.db_session import Base


class ShipVoyageRiskMap(Base):
  __tablename__ = "ship_voyage_risk_map"
  __table_args__ = {'schema': 'gateway'}

  ship_id: Mapped[str] = mapped_column(primary_key=True)
  request_time: Mapped[datetime] = mapped_column(primary_key=True)
  prediction_route_type: Mapped[str] = mapped_column(primary_key=True)
  zone_order: Mapped[int] = mapped_column()

  zone_entry_time: Mapped[datetime] = mapped_column()
  zone_exit_time: Mapped[datetime] = mapped_column()

  longitude: Mapped[float] = mapped_column(Float)
  latitude: Mapped[float] = mapped_column(Float)

  voyage_risk_map_id: Mapped[str] = mapped_column(String)
  longitude_length: Mapped[float] = mapped_column(Float)
  latitude_length: Mapped[float] = mapped_column(Float)
  wind_speed: Mapped[float] = mapped_column(Float)
  wind_direction: Mapped[float] = mapped_column(Float)
  wave_height: Mapped[float] = mapped_column(Float)
  wave_direction: Mapped[float] = mapped_column(Float)
  relative_humidity: Mapped[float] = mapped_column(Float)

  total_column_water_vapor: Mapped[float] = mapped_column(Float)
  skin_temperature: Mapped[float] = mapped_column(Float)
  voyage_risk_level: Mapped[float] = mapped_column(Float)

  tidal_current_speed: Mapped[float] = mapped_column(Float)
  tidal_current_direction: Mapped[float] = mapped_column(Float)

  typhoon_longitude: Mapped[float] = mapped_column(Float)
  typhoon_latitude: Mapped[float] = mapped_column(Float)
  typhoon_grade: Mapped[float] = mapped_column(Float)
  typhoon_speed: Mapped[float] = mapped_column(Float)
