"""Domain node and API models for the root System entity."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field

from gosmart_dictionary_service.model.auditable import AuditableNode

DEFAULT_SYSTEM_NAME = "GoSmart Health Dictionary Service"
DEFAULT_SYSTEM_OWNER = "GoSmart Health"
DEFAULT_SYSTEM_ADDRESS = "Half Moon Bay, California"
DEFAULT_SYSTEM_EMAIL = "admin@gosmart.health"
DEFAULT_SYSTEM_PHONE = "+1 (650) 535-0097"
DEFAULT_SYSTEM_SERIAL_NUMBER = "GSH-DS-0001"


class SystemNode(AuditableNode):
    """Auditable graph node representing the root GoSmart DS platform instance (``:System``)."""

    __label__: ClassVar[str] = "System"

    name: str = Field(default=DEFAULT_SYSTEM_NAME, description="Name of the system")
    owner: str = Field(default=DEFAULT_SYSTEM_OWNER, description="Owner user or organization")
    address: str = Field(default=DEFAULT_SYSTEM_ADDRESS, description="Physical address string")
    email: str = Field(default=DEFAULT_SYSTEM_EMAIL, description="Contact email address (non-auth)")
    phone: str = Field(default=DEFAULT_SYSTEM_PHONE, description="Contact phone number (non-auth)")
    serial_number: str = Field(default=DEFAULT_SYSTEM_SERIAL_NUMBER, description="System serial number")
    installed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Installation UTC timestamp",
    )

    def get_identity_subset(self) -> Dict[str, Any]:
        subset: Dict[str, Any] = {
            "entity": self.__label__,
            "serial_number": self.serial_number.strip(),
        }
        if self.test_id:
            subset["test_id"] = self.test_id
        return subset

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Serialize model fields into Neo4j-compatible primitives."""
        data = self.model_dump()
        for key, val in list(data.items()):
            if isinstance(val, datetime):
                data[key] = val.isoformat()
        return data


class SystemResponse(BaseModel):
    """API response model for the root System node."""

    id: str = Field(..., description="Deterministic System ID")
    name: str = Field(..., description="Name of the system")
    owner: str = Field(..., description="Owner user or organization")
    address: str = Field(..., description="Physical address string")
    email: str = Field(..., description="Contact email address (non-auth)")
    phone: str = Field(..., description="Contact phone number (non-auth)")
    serial_number: str = Field(..., description="System serial number")
    installed_at: datetime = Field(..., description="Installation UTC DateTime")
    test_id: Optional[str] = Field(default=None, description="Test identifier if ephemeral")
    action: Optional[str] = Field(default="CREATE", description="Auditable action state")
    created_at: Optional[datetime] = Field(default=None, description="Created timestamp (UTC)")
    updated_at: Optional[datetime] = Field(default=None, description="Updated timestamp (UTC)")

    model_config = {"populate_by_name": True}


class SystemUpdateRequest(BaseModel):
    """API request model to update mutable properties of the System node."""

    name: Optional[str] = Field(default=None, description="Name of the system")
    owner: Optional[str] = Field(default=None, description="Owner user or organization")
    address: Optional[str] = Field(default=None, description="Physical address string")
    email: Optional[str] = Field(default=None, description="Contact email address (non-auth)")
    phone: Optional[str] = Field(default=None, description="Contact phone number (non-auth)")
    serial_number: Optional[str] = Field(default=None, description="System serial number")
    installed_at: Optional[datetime] = Field(default=None, description="Installation UTC DateTime")


class SystemBootstrapResponse(BaseModel):
    """API / service response model for system, root group, and default LLM_Config bootstrapping."""

    success: bool = Field(..., description="Whether bootstrap succeeded")
    message: str = Field(..., description="Bootstrap status message")
    system: SystemResponse = Field(..., description="System node representation")
    root_group: Dict[str, Any] = Field(default_factory=dict, description="Top-level GoSmart Health Group node")
    llm_configs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sanitized default LLM_Config entries seeded on the root Group",
    )


class TestCleanupResponse(BaseModel):
    """API / service response model for scoped test data cleanup."""

    success: bool = Field(..., description="Whether cleanup succeeded")
    message: str = Field(..., description="Status message")
    total_deleted: int = Field(default=0, description="Total count of nodes detached and deleted")
    deleted_by_label: Dict[str, int] = Field(default_factory=dict, description="Breakdown of deleted nodes by label")
    labels_cleaned: List[str] = Field(default_factory=list, description="List of labels scanned and cleaned")
    test_id: Optional[str] = Field(default=None, description="Test identifier that was cleaned up")


__all__ = [
    "DEFAULT_SYSTEM_NAME",
    "DEFAULT_SYSTEM_OWNER",
    "DEFAULT_SYSTEM_ADDRESS",
    "DEFAULT_SYSTEM_EMAIL",
    "DEFAULT_SYSTEM_PHONE",
    "DEFAULT_SYSTEM_SERIAL_NUMBER",
    "SystemNode",
    "SystemResponse",
    "SystemUpdateRequest",
    "SystemBootstrapResponse",
    "TestCleanupResponse",
]
