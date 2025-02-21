from sqlalchemy.orm import mapped_column, Mapped
from app.config.db_session import Base

from decimal import Decimal

class SatelliteExtractionWind(Base):
  __tablename__ = "satellite_extraction_wind"
  __table_args__ = {'schema': 'gateway'}

  satellite_eo_image_id: Mapped[str] = mapped_column(primary_key=True)
  longitude: Mapped[Decimal] = mapped_column(primary_key=True)
  latitude: Mapped[Decimal] = mapped_column(primary_key=True)
  longitude_length: Mapped[float] = mapped_column(Float)
  latitude_length: Mapped[float] = mapped_column(Float)
  wind_speed: Mapped[float] = mapped_column(Float)
  wind_direction: Mapped[float] = mapped_column(Float)
