from datetime import datetime
from sqlalchemy import String, Numeric, Float
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import mapped_column, Mapped
from app.config.db_session import Base


class ShipStatusHist(Base):
  __tablename__ = "ship_status_hist"
  __table_args__ = {'schema': 'gateway'}

  ship_id: Mapped[str] = mapped_column(String(64), primary_key=True)
  status_time: Mapped[datetime] = mapped_column(primary_key=True)
  longitude: Mapped[Decimal] = mapped_column(nullable=False)
  latitude: Mapped[Decimal] = mapped_column(nullable=False)
  cog: Mapped[float] = mapped_column(Float)
  sog: Mapped[float] = mapped_column(Float)
