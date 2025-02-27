from datetime import datetime
from sqlalchemy.orm import Session

from app.models.ais_ship_data_hist import AisShipDataHist

def get_ship_data_history(
    db: Session,
    timestamp_start: datetime,
    timestamp_end: datetime,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float
) -> List[AisShipDataLast]:
    return db.query(
        AisShipDataLast.mmsi,
        AisShipDataLast.timestamp,
        AisShipDataLast.lo_lo.label('longitude'),
        AisShipDataLast.la_la.label('latitude'),
        AisShipDataLast.sog,
        AisShipDataLast.cog,
        AisShipDataLast.ship_nm,
        AisShipDataLast.ship_type,
        AisShipDataLast.ship_dim_a,
        AisShipDataLast.ship_dim_b,
        AisShipDataLast.ship_dim_c,
        AisShipDataLast.ship_dim_d,
        AisShipDataLast.nvg_sttus
    ).filter(
        AisShipDataLast.timestamp.between(timestamp_start, timestamp_end),
        AisShipDataLast.lo_lo.between(lon_min, lon_max),
        AisShipDataLast.la_la.between(lat_min, lat_max)
    ).all()