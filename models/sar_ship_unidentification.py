from sqlalchemy.orm import mapped_column, Mapped
from app.config.db_session import Base


class SarShipUnidentification(Base):
  __tablename__ = "sar_ship_unidentification"
  __table_args__ = {'schema': 'gateway'}

  satellite_sar_image_id: Mapped[str] = mapped_column(primary_key=True)
  unidentification_ship_id: Mapped[str] = mapped_column(primary_key=True)
  longitude: Mapped[float] = mapped_column(Float, nullable=False)
  latitude: Mapped[float] = mapped_column(Float, nullable=False)
  prediction_cog: Mapped[float] = mapped_column(Float)
  prediction_sog: Mapped[float] = mapped_column(Float)
  prediction_ship_type: Mapped[str] = mapped_column()
  prediction_length: Mapped[float] = mapped_column(Float)
  prediction_width: Mapped[float] = mapped_column(Float)
