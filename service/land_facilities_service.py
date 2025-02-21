#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sqlalchemy.orm import Session

from app.models.land_facilities import LandFacilities

# --------------------------------------------------------------------------------------------
## 데이터 조회
def get_land_facilities(db: Session, id: str) -> List[LandFacilities]:
    return db.query(LandFacilities).filter(LandFacilities.facility_id == facility_id).all()
