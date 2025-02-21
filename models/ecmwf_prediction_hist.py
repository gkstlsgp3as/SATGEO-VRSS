from datetime import datetime

from sqlalchemy.orm import mapped_column, Mapped

from app.config.db_session import Base
from decimal import Decimal


class EcmwfPredictionHist(Base):
    __tablename__ = "ecmwf_prediction_hist"
    __table_args__ = {'schema': 'gateway'}

    prediction_standard_time: Mapped[datetime] = mapped_column(primary_key=True)
    prediction_time: Mapped[datetime] = mapped_column(primary_key=True)
    longitude: Mapped[Decimal] = mapped_column(primary_key=True)
    latitude: Mapped[Decimal] = mapped_column(primary_key=True)
    longitude_length: Mapped[float] = mapped_column(Float)
    latitude_length: Mapped[float] = mapped_column(Float)
    wind_speed_real: Mapped[float] = mapped_column(Float)
    wind_direction_real: Mapped[float] = mapped_column(Float)
    wave_height_real: Mapped[float] = mapped_column(Float)
    wave_direction_real: Mapped[float] = mapped_column(Float)
    relative_humidity: Mapped[float] = mapped_column(Float)
    total_column_water_vapor: Mapped[float] = mapped_column(Float)
    skin_temperature: Mapped[float] = mapped_column(Float)
