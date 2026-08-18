from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime, timezone
try:
    from backend.database import Base
except ModuleNotFoundError:
    from database import Base

class SafetyEvent(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    worker_id = Column(String, nullable=True, index=True)
    event_type = Column(String, nullable=False, index=True)
    track_id = Column(Integer, nullable=False, index=True)
    severity = Column(String, nullable=False)
    zone = Column(String, nullable=True)
    confidence = Column(Float, nullable=False)
    timestamp = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class EmergencyEvent(Base):
    __tablename__ = "emergency_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    event_type = Column(String, nullable=False, index=True)  # "SOS" or "IMU_FALL"
    worker_id = Column(String, nullable=False, index=True)
    band_id = Column(String, nullable=False)
    battery = Column(Integer, nullable=False, default=100)
    rssi = Column(Integer, nullable=False, default=-65)
    zone = Column(String, nullable=True)
    status = Column(String, nullable=False, default="open", index=True)  # "open" or "resolved"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class BandHeartbeat(Base):
    __tablename__ = "band_heartbeats"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    worker_id = Column(String, nullable=False, index=True)
    band_id = Column(String, nullable=False, index=True)
    battery = Column(Integer, nullable=False, default=100)
    rssi = Column(Integer, nullable=False, default=-65)
    zone = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

