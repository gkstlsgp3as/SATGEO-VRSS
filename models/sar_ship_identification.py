from sqlalchemy.orm import mapped_column, Mapped
from app.config.db_session import Base


class SarShipIdentification(Base):
  __tablename__ = "sar_ship_identification"
  __table_args__ = {'schema': 'gateway'}

  satellite_sar_image_id: Mapped[str] = mapped_column(primary_key=True)
  identification_ship_id: Mapped[str] = mapped_column(primary_key=True)
  longitude: Mapped[Decimal] = mapped_column(nullable=False)
  latitude: Mapped[Decimal] = mapped_column(nullable=False)
  interpolation_cog: Mapped[float] = mapped_column(Float)
  interpolation_sog: Mapped[float] = mapped_column(Float)
  type: Mapped[str] = mapped_column()
  end: Mapped[str] = mapped_column()
  detection_yn: Mapped[bool] = mapped_column()
  detection_longitude: Mapped[float] = mapped_column(Float)
  detection_latitude: Mapped[float] = mapped_column(Float)
