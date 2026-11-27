"""Health check endpoint for GoSmart DS."""

from fastapi import APIRouter, Depends

from gosmart_dictionary_service.core.config import Settings, get_settings
from gosmart_dictionary_service.db.neo4j import Neo4jClient, get_neo4j
from gosmart_dictionary_service.model.health import HealthResponse, ServiceHealth

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def check_health(
    settings: Settings = Depends(get_settings),
    neo4j: Neo4jClient = Depends(get_neo4j),
) -> HealthResponse:
    """Return service and Neo4j connectivity health status."""
    neo4j_healthy = await neo4j.verify_connectivity()
    services = {
        "neo4j": ServiceHealth(
            name="Neo4j Graph Database",
            status="HEALTHY" if neo4j_healthy else "UNHEALTHY",
            details={"connection_uri": settings.DB_CONNECTION_URI},
        )
    }

    overall_status = "HEALTHY" if neo4j_healthy else "DEGRADED"

    return HealthResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        services=services,
        db_connection_uri=settings.DB_CONNECTION_URI,
    )
