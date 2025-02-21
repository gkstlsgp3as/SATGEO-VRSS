from datetime import datetime
from sqlalchemy import String, TIMESTAMP
from sqlalchemy.orm import mapped_column, Mapped
from app.config.db_session import Base


class InterestShip(Base):
  __tablename__ = "interest_ship"
  __table_args__ = {'schema': 'gateway'}

  interest_ship_id: Mapped[str] = mapped_column(String(64), primary_key=True)
  group_id: Mapped[str] = mapped_column(String(36))
  interest_ship_type: Mapped[str] = mapped_column(String(36), nullable=False)
