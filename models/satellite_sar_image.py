from datetime import datetime
from sqlalchemy import String, Float, Text, JSON
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import mapped_column, Mapped
from app.config.db_session import Base


class SatelliteSarImage(Base):
  __tablename__ = "satellite_sar_image"
  __table_args__ = {'schema': 'gateway'}

  satellite_sar_image_id: Mapped[str] = mapped_column(primary_key=True)
  photography_start_time: Mapped[datetime] = mapped_column()
  photography_end_time: Mapped[datetime] = mapped_column()
  image_resolution_type: Mapped[str] = mapped_column()
  image_polygon: Mapped[dict] = mapped_column(JSON)
  image_size: Mapped[float] = mapped_column(Float)
  image_path: Mapped[str] = mapped_column()
  description: Mapped[str] = mapped_column(Text)
