"""
Database Models and Schema
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, LargeBinary,text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
from core.config import settings

Base = declarative_base()

# ==================== MODELS ====================

class Camera(Base):
    """Camera configuration"""
    __tablename__ = "cameras"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, unique=True, index=True)
    name = Column(String)
    location = Column(String)
    stream_url = Column(String)  # RTSP/HTTP URL
    camera_type = Column(String)  # IP/WiFi/USB
    status = Column(String, default="active")  # active/inactive/error
    fps = Column(Integer, default=30)
    resolution = Column(String, default="1920x1080")
    zone_config = Column(Text)  # JSON string for zone boundaries
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    detections = relationship(
    "Detection",
    back_populates="camera",
    cascade="all, delete-orphan")

    recordings = relationship(
    "Recording",
    back_populates="camera",
    cascade="all, delete-orphan")

class Person(Base):
    """Person records with face embeddings"""
    __tablename__ = "persons"
    
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=True)
    classification = Column(String)  # employee/owner/unknown/visitor
    face_embedding = Column(LargeBinary)  # Stored as bytes
    face_image_path = Column(String)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    total_appearances = Column(Integer, default=1)
    notes = Column(Text, nullable=True)
    
    # Relationships
    time_logs = relationship("TimeLog", back_populates="person")
    detections = relationship("Detection", back_populates="person")

class Detection(Base):
    """Real-time person detections"""
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"))
    camera_label = Column(String, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)
    detection_uuid = Column(String, unique=True, index=True)
    
    # Detection info
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)
    confidence = Column(Float)
    
    # Action detection
    action = Column(String, nullable=True,index=True)
    action_confidence = Column(Float, nullable=True)
    
    # Zone information
    zone = Column(Integer,index=True)  # 1, 2, or 3
    distance_from_camera = Column(Float)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    frame_path = Column(String, nullable=True)
    
    # Face image & person info (added by migration)
    face_image_url        = Column(String(512), nullable=True)
    person_string_id      = Column(String(32), nullable=True)
    person_name           = Column(String(128), nullable=True)
    person_classification = Column(String(32), nullable=True)
    
    # Relationships
    camera = relationship("Camera", back_populates="detections")
    person = relationship("Person", back_populates="detections")

class TimeLog(Base):
    """Person entry/exit time tracking"""
    __tablename__ = "time_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"))
    
    entry_time = Column(DateTime, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    duration = Column(Integer, nullable=True)  # in seconds
    entry_camera = Column(String)
    exit_camera = Column(String, nullable=True)
    
    date = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    person = relationship("Person", back_populates="time_logs")

class Alert(Base):
    """Security alerts"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String, unique=True, index=True)
    
    # Alert details
    alert_type = Column(String)  # fighting, weapon, loitering, etc.
    alert_level = Column(Integer,index=True)  # 1 (critical), 2 (medium), 3 (normal)
    severity = Column(String)  # critical/high/medium/low
    
    # Location
    camera_id = Column(Integer, ForeignKey("cameras.id"))
    zone = Column(Integer,index=True)
    
    # Description
    description = Column(Text)
    action_detected = Column(String)
    
    # Media
    video_path = Column(String, nullable=True)
    snapshot_path = Column(String, nullable=True)
    
    # Status
    status = Column(String, default="active",index=True)  # active/acknowledged/resolved
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Notifications sent
    email_sent = Column(Boolean, default=False)
    sms_sent = Column(Boolean, default=False)
    whatsapp_sent = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

class Recording(Base):
    """Continuous recording sessions"""
    __tablename__ = "recordings"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    camera_label = Column(String, index=True)
    
    file_path = Column(String)
    file_size = Column(Integer)  # in bytes
    duration = Column(Integer)  # in seconds
    
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    
    fps = Column(Integer)
    resolution = Column(String)
    
    # Relationships
    camera = relationship("Camera", back_populates="recordings")

class UnknownPerson(Base):
    """Track unknown persons for clustering"""
    __tablename__ = "unknown_persons"
    
    id = Column(Integer, primary_key=True, index=True)
    
    face_embedding = Column(LargeBinary)
    face_image_path = Column(String)
    
    camera_id = Column(Integer, ForeignKey("cameras.id"))
    detection_time = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Clustering info
    cluster_id = Column(Integer, nullable=True)
    is_clustered = Column(Boolean, default=False)
    
    # Alert status
    alert_sent = Column(Boolean, default=False)

class SystemLog(Base):
    """System activity logs"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    log_type = Column(String)  # info/warning/error/critical
    category = Column(String)  # camera/ai/alert/system
    message = Column(Text)
    details = Column(Text, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

# ==================== DATABASE INITIALIZATION ====================

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_pre_ping=True,
    connect_args={
        "check_same_thread": False,
        "timeout": 30
    }
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def init_db():
    async with engine.begin() as conn:

        try:
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA synchronous=NORMAL"))
            await conn.execute(text("PRAGMA cache_size=-64000"))
            await conn.execute(text("PRAGMA temp_store=MEMORY"))
        except:
            pass

        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
