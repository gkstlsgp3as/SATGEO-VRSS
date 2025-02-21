from sqlalchemy import String, Numeric, Integer, Text, JSON
from sqlalchemy.orm import mapped_column, Mapped
from geoalchemy2 import Geometry
from app.config.db_session import Base


class InterestArea(Base):
  __tablename__ = "interest_area"
  __table_args__ = {'schema': 'gateway'}

  interest_area_id: Mapped[str] = mapped_column(primary_key=True)
  interest_area_name: Mapped[str] = mapped_column()
  group_id: Mapped[str] = mapped_column()
  shape_type: Mapped[str] = mapped_column()

  geo_data: Mapped[str] = mapped_column(Text)
  geo_data_geom: Mapped[str] = mapped_column(Geometry("GEOMETRY"))

  longitude: Mapped[float] = mapped_column(Float)
  latitude: Mapped[float] = mapped_column(Float)
  radius_lt: Mapped[int] = mapped_column()

  route: Mapped[dict] = mapped_column(JSON)
  route_geom: Mapped[str] = mapped_column(Geometry("GEOMETRY"))
