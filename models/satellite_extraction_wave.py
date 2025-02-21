from sqlalchemy.orm import mapped_column, Mapped

from app.config.db_session import Base
from decimal import Decimal

class SatelliteExtractionWave(Base):
    __tablename__ = "satellite_extraction_wave"
    __table_args__ = {'schema': 'gateway'}

    satellite_eo_image_id: Mapped[str] = mapped_column(primary_key=True)
    longitude: Mapped[Decimal] = mapped_column(primary_key=True)
    latitude: Mapped[Decimal] = mapped_column(primary_key=True)
    longitude_length: Mapped[float] = mapped_column(Float)
    latitude_length: Mapped[float] = mapped_column(Float)
    wave_length: Mapped[float] = mapped_column(Float)
    azimuth_cutoff: Mapped[float] = mapped_column(Float)
    mean_wave_height: Mapped[float] = mapped_column(Float)
    mean_wave_direction: Mapped[float] = mapped_column(Float)
