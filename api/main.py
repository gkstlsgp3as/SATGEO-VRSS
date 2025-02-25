from fastapi import APIRouter

from app.api.routers import ships, risk_mappings 

api_router = APIRouter()
api_router.include_router(ships.router, prefix="/ships", tags=["S_ship"])
api_router.include_router(risk_mappings.router, prefix="/risk-mappings", tags=["W_riskmapping"])
