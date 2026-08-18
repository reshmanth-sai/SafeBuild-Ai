from fastapi import FastAPI, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timezone

try:
    from backend.database import Base, engine, get_db
    from backend.models import SafetyEvent, EmergencyEvent, BandHeartbeat
    from backend.schemas import (
        SafetyEventCreate, SafetyEventResponse,
        EmergencyEventCreate, EmergencyEventResponse,
        HeartbeatCreate, HeartbeatResponse, BandStatusResponse
    )
except ModuleNotFoundError:
    from database import Base, engine, get_db
    from models import SafetyEvent, EmergencyEvent, BandHeartbeat
    from schemas import (
        SafetyEventCreate, SafetyEventResponse,
        EmergencyEventCreate, EmergencyEventResponse,
        HeartbeatCreate, HeartbeatResponse, BandStatusResponse
    )

from fastapi.middleware.cors import CORSMiddleware

# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SafeBuild AI Backend",
    description="Safety Event Ingestion & Monitoring API",
    version="1.0.0"
)

# Note: allow_origins=["*"] is used for local prototype demo only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "SafeBuild AI Backend API is running."}

@app.post("/events/safety", response_model=SafetyEventResponse, status_code=status.HTTP_201_CREATED)
def create_safety_event(event_in: SafetyEventCreate, db: Session = Depends(get_db)):
    """
    Ingest a new safety violation event from the CV detection pipeline.
    """
    db_event = SafetyEvent(
        worker_id=event_in.worker_id,
        event_type=event_in.event_type,
        track_id=event_in.track_id,
        severity=event_in.severity,
        zone=event_in.zone,
        confidence=event_in.confidence,
        timestamp=event_in.timestamp,
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event

@app.get("/events/safety", response_model=List[SafetyEventResponse])
def get_safety_events(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    """
    Retrieve the most recent N safety events, ordered newest-first.
    """
    events = db.query(SafetyEvent).order_by(SafetyEvent.id.desc()).limit(limit).all()
    return events

def _to_utc_iso(ts):
    if not ts:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    if isinstance(ts, str):
        if not ts.endswith("Z") and "+" not in ts and "-" not in ts[10:]:
            return ts + "+00:00"
        return ts
    return str(ts)

@app.get("/events/timeline/{worker_id}")
def get_worker_timeline(worker_id: str, limit: int = Query(50, ge=1, le=100), db: Session = Depends(get_db)):
    """
    Retrieve unified chronological incident timeline for a specific worker.
    Merges both CV safety events and Wearable emergency events into a single sorted feed (newest-first).
    """
    safety_records = db.query(SafetyEvent).filter(SafetyEvent.worker_id == worker_id).all()
    emergency_records = db.query(EmergencyEvent).filter(EmergencyEvent.worker_id == worker_id).all()

    timeline = []

    for s in safety_records:
        ts_str = _to_utc_iso(s.timestamp if s.timestamp else s.created_at)
        timeline.append({
            "id": s.id,
            "source": "safety",
            "event_type": s.event_type,
            "worker_id": s.worker_id,
            "track_id": s.track_id,
            "band_id": None,
            "severity": s.severity,
            "confidence": s.confidence,
            "battery": None,
            "rssi": None,
            "zone": s.zone,
            "status": None,
            "timestamp": ts_str
        })

    for e in emergency_records:
        ts_str = _to_utc_iso(e.created_at)
        timeline.append({
            "id": e.id,
            "source": "emergency",
            "event_type": e.event_type,
            "worker_id": e.worker_id,
            "track_id": None,
            "band_id": e.band_id,
            "severity": "HIGH",
            "confidence": 1.0,
            "battery": e.battery,
            "rssi": e.rssi,
            "zone": e.zone,
            "status": e.status,
            "timestamp": ts_str
        })

    # Sort combined timeline newest-first
    timeline.sort(key=lambda x: x["timestamp"], reverse=True)
    return timeline[:limit]

# ---------------------------------------------------------------------------
# EMERGENCY & WEARABLE ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/events/emergency", response_model=EmergencyEventResponse, status_code=status.HTTP_201_CREATED)
def create_emergency_event(event_in: EmergencyEventCreate, db: Session = Depends(get_db)):
    """
    Ingest a new emergency SOS / IMU fall event from wearable band simulator.
    """
    db_event = EmergencyEvent(
        event_type=event_in.event_type,
        worker_id=event_in.worker_id,
        band_id=event_in.band_id,
        battery=event_in.battery,
        rssi=event_in.rssi,
        zone=event_in.zone,
        status="open"
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event

@app.get("/events/emergency", response_model=List[EmergencyEventResponse])
def get_emergency_events(
    limit: int = Query(50, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    db: Session = Depends(get_db)
):
    """
    Retrieve recent emergency events, ordered newest-first. Optionally filter by status ('open' or 'resolved').
    """
    query = db.query(EmergencyEvent)
    if status_filter:
        query = query.filter(EmergencyEvent.status == status_filter)
    events = query.order_by(EmergencyEvent.id.desc()).limit(limit).all()
    return events

@app.patch("/events/emergency/{event_id}/resolve", response_model=EmergencyEventResponse)
def resolve_emergency_event(event_id: int, db: Session = Depends(get_db)):
    """
    Mark an active emergency event as resolved by rescue responders.
    """
    event = db.query(EmergencyEvent).filter(EmergencyEvent.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail=f"Emergency event #{event_id} not found.")
    
    event.status = "resolved"
    db.commit()
    db.refresh(event)
    return event


# ---------------------------------------------------------------------------
# HEARTBEAT & LIVENESS ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/events/heartbeat", response_model=HeartbeatResponse, status_code=status.HTTP_201_CREATED)
def create_heartbeat(hb_in: HeartbeatCreate, db: Session = Depends(get_db)):
    """
    Ingest a periodic liveness heartbeat packet from a smart safety band.
    """
    db_hb = BandHeartbeat(
        worker_id=hb_in.worker_id,
        band_id=hb_in.band_id,
        battery=hb_in.battery,
        rssi=hb_in.rssi,
        zone=hb_in.zone
    )
    db.add(db_hb)
    db.commit()
    db.refresh(db_hb)
    return db_hb


@app.get("/bands/status", response_model=List[BandStatusResponse])
def get_bands_status(db: Session = Depends(get_db)):
    """
    Retrieve live liveness/heartbeat status for all known wearable bands.
    Combines latest heartbeat and emergency event timestamps to determine
    if a band is ONLINE (<=10s), WARNING (10s-30s), or SIGNAL LOST (>30s).
    """
    heartbeats = db.query(BandHeartbeat).all()
    emergencies = db.query(EmergencyEvent).all()

    now = datetime.now(timezone.utc)
    band_map = {}

    def _ensure_utc(dt):
        if dt is None:
            return now
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    for hb in heartbeats:
        bid = hb.band_id
        dt = _ensure_utc(hb.created_at)
        if bid not in band_map or dt > band_map[bid]["time"]:
            band_map[bid] = {
                "band_id": bid,
                "worker_id": hb.worker_id,
                "battery": hb.battery,
                "rssi": hb.rssi,
                "zone": hb.zone,
                "time": dt,
                "type": "HEARTBEAT"
            }

    for em in emergencies:
        bid = em.band_id
        dt = _ensure_utc(em.created_at)
        if bid not in band_map or dt > band_map[bid]["time"]:
            band_map[bid] = {
                "band_id": bid,
                "worker_id": em.worker_id,
                "battery": em.battery,
                "rssi": em.rssi,
                "zone": em.zone,
                "time": dt,
                "type": em.event_type
            }

    result = []
    for bid, data in sorted(band_map.items()):
        elapsed = max(0.0, (now - data["time"]).total_seconds())
        if elapsed <= 10.0:
            lbl = "ONLINE"
            clr = "green"
        elif elapsed <= 30.0:
            lbl = "WARNING"
            clr = "amber"
        else:
            lbl = "SIGNAL LOST"
            clr = "red"

        result.append(BandStatusResponse(
            band_id=data["band_id"],
            worker_id=data["worker_id"],
            battery=data["battery"],
            rssi=data["rssi"],
            zone=data["zone"],
            last_signal_time=data["time"],
            last_signal_type=data["type"],
            last_seen_seconds=round(elapsed, 1),
            status_label=lbl,
            status_color=clr
        ))

    return result


