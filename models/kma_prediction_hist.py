from datetime import datetime

from sqlalchemy.orm import mapped_column, Mapped

from app.config.db_session import Base
from decimal import Decimal

class KmaPredictionHist(Base):
  __tablename__ = "kma_prediction_hist"
  __table_args__ = {'schema': 'gateway'}

  prediction_standard_time: Mapped[datetime] = mapped_column(primary_key=True)
  prediction_time: Mapped[datetime] = mapped_column(primary_key=True)
  longitude: Mapped[Decimal] = mapped_column(primary_key=True)
  latitude: Mapped[Decimal] = mapped_column(primary_key=True)
  wind_u_component: Mapped[float] = mapped_column(Float)
  wind_v_component: Mapped[float] = mapped_column(Float)
  large_scale_precipitation: Mapped[float] = mapped_column(Float)
  specific_humidity: Mapped[float] = mapped_column(Float)
  relative_humidity: Mapped[float] = mapped_column(Float)
