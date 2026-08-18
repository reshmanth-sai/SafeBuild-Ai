from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class SafetyEventCreate(BaseModel):
    worker_id: Optional[str] = Field(None, description="Worker identifier, e.g. W001")
    event_type: str = Field(..., description="One of: MISSING_HELMET, MISSING_VEST, ZONE_INTRUSION, FALL_DETECTED")
    track_id: int = Field(..., description="ByteTrack track ID")
    severity: str = Field(..., description="Severity level: HIGH, MEDIUM, LOW")
    confidence: float = Field(..., description="Model detection confidence (0.0 to 1.0)")
    zone: Optional[str] = Field(None, description="Zone name if applicable")
    timestamp: str = Field(..., description="ISO8601 timestamp string")

class SafetyEventResponse(SafetyEventCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class TimelineItemResponse(BaseModel):
    id: int
    source: str  # "safety" or "emergency"
    event_type: str
    worker_id: Optional[str] = None
    track_id: Optional[int] = None
    band_id: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    battery: Optional[int] = None
    rssi: Optional[int] = None
    zone: Optional[str] = None
    status: Optional[str] = None
    timestamp: str

    class Config:
        from_attributes = True

class EmergencyEventCreate(BaseModel):
    worker_id: str = Field(..., description="Worker identifier, e.g. W001")
    band_id: str = Field(..., description="Wearable band ID, e.g. B001")
    event_type: str = Field(..., description="Event type: SOS or IMU_FALL")
    battery: int = Field(78, description="Battery percentage (0 to 100)")
    rssi: int = Field(-67, description="Signal strength in dBm")
    zone: Optional[str] = Field(None, description="Current zone location")

class EmergencyEventResponse(EmergencyEventCreate):
    id: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True
