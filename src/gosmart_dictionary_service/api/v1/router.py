"""Top-level API v1 router for GoSmart DS."""

from fastapi import APIRouter

from gosmart_dictionary_service.api.v1.endpoints.health import router as health_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router)
