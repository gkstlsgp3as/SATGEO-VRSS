from datetime import datetime
from sqlalchemy import String, Float, JSON
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import mapped_column, Mapped
from app.config.db_session import Base


class SatelliteEoImage(Base):
  __tablename__ = "satellite_eo_image"
  __table_args__ = {'schema': 'gateway'}

  satellite_eo_image_id: Mapped[str] = mapped_column(String(32), primary_key=True)
  photography_start_time: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False)
  photography_end_time: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False)
  image_resolution: Mapped[str] = mapped_column(String(20), nullable=False)
  output_resolution_type: Mapped[str] = mapped_column(String(20), nullable=False)

  image_polygon: Mapped[dict] = mapped_column(JSON)
  satellite_image_type: Mapped[str] = mapped_column(String(4))
  image_size: Mapped[float] = mapped_column(Float)
  image_path: Mapped[str] = mapped_column(String(1024))
  description: Mapped[str] = mapped_column(String)
