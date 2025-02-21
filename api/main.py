from fastapi import APIRouter

from app.api.routers import ships, risk_mappings, 

api_router = APIRouter()
api_router.include_router(ships.router, prefix="/ships", tags=["S_ship"])
api_router.include_router(risk_mappings.router, prefix="/risk-mappings", tags=["W_riskmapping"])
api_router.include_router(coastal_monitorings.router, prefix="/coastal-monitorings", tags=["M_coastalmonitoring"])
api_router.include_router(new_facilities.router, prefix="/new-facilities", tags=["N_newfacility"])
