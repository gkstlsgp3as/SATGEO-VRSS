from datetime import datetime

from sqlalchemy.orm import mapped_column, Mapped

from app.config.db_session import Base
from decimal import Decimal


class VoyageRiskMapLast(Base):
    __tablename__ = "voyage_risk_map_last"
    # __table_args__ = {'schema': 'public'}
    __table_args__ = {'schema': 'gateway'}

    longitude: Mapped[Decimal] = mapped_column(primary_key=True)
    latitude: Mapped[Decimal] = mapped_column(primary_key=True)
    voyage_risk_map_id: Mapped[str] = mapped_column()
    longitude_length: Mapped[float] = mapped_column(Float)
    latitude_length: Mapped[float] = mapped_column(Float)
    wind_speed: Mapped[float] = mapped_column(Float)
    wind_direction: Mapped[float] = mapped_column(Float)
    wave_height: Mapped[float] = mapped_column(Float)
    wave_direction: Mapped[float] = mapped_column(Float)
    relative_humidity: Mapped[float] = mapped_column(Float)
    total_column_water_vapor: Mapped[float] = mapped_column(Float)
    skin_temperature: Mapped[float] = mapped_column(Float)
    tidal_current_speed: Mapped[float] = mapped_column(Float)
    tidal_current_direction: Mapped[float] = mapped_column(Float)
    typhoon_longitude: Mapped[float] = mapped_column(Float)
    typhoon_latitude: Mapped[float] = mapped_column(Float)
    typhoon_grade: Mapped[float] = mapped_column(Float)
