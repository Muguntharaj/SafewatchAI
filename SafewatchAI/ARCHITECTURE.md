# 🏗️ SafeWatch AI - System Architecture

**Powered by Neelaminds Private Limited**

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Technology Stack](#technology-stack)
3. [Architecture Diagram](#architecture-diagram)
4. [AI Models](#ai-models)
5. [Backend Components](#backend-components)
6. [Frontend Components](#frontend-components)
7. [Data Flow](#data-flow)
8. [Database Schema](#database-schema)
9. [API Architecture](#api-architecture)
10. [Security Features](#security-features)

---

## 🎯 System Overview

SafeWatch AI is a comprehensive AI-powered CCTV surveillance system that provides:

- **Real-time person detection** using YOLO v8
- **Action recognition** using CLIP model
- **Face recognition** using FaceNet
- **Multi-camera support** (IP/WiFi/USB)
- **3-level alert system**
- **Multi-channel notifications** (Email/SMS/WhatsApp)
- **Continuous recording** with automatic segmentation
- **Face clustering** for unknown persons
- **Zone-based detection**
- **CPU/GPU compatibility**

---

## 💻 Technology Stack

### Backend (Python)
- **Framework**: FastAPI 0.109.2
- **Server**: Uvicorn
- **Database**: SQLite + SQLAlchemy (async)
- **AI/ML**:
  - PyTorch 2.2.0
  - Ultralytics YOLO v8
  - Transformers (CLIP)
  - FaceNet-PyTorch
- **Video Processing**: OpenCV, FFmpeg
- **Messaging**: SMTP, Twilio, PyWhatKit

### Frontend (React + TypeScript)
- **Framework**: React 18.3.1
- **Build Tool**: Vite 6.3.5
- **Styling**: Tailwind CSS 4.0
- **UI Components**: Radix UI, shadcn/ui
- **Charts**: Recharts 2.15.2
- **Animations**: Motion (Framer Motion) 12.23.24
- **State Management**: React Hooks
- **HTTP Client**: Fetch API
- **WebSocket**: Native WebSocket API

### Infrastructure
- **OS**: Cross-platform (Windows/Linux/macOS)
- **CUDA**: Optional GPU acceleration
- **Storage**: Local filesystem
- **Networking**: HTTP/HTTPS, WebSocket, RTSP

---

## 🏛️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND LAYER                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  React App (TypeScript)                                  │  │
│  │  ├── Dashboard                                           │  │
│  │  ├── Analytics (Charts)                                  │  │
│  │  ├── Camera Grid                                         │  │
│  │  ├── Person Classifier                                   │  │
│  │  ├── Alert Manager                                       │  │
│  │  └── Time Logs                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↕ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────────┐
│                          API LAYER                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  FastAPI (Python)                                        │  │
│  │  ├── /api/cameras       - Camera management             │  │
│  │  ├── /api/detections    - Detection queries             │  │
│  │  ├── /api/alerts        - Alert management              │  │
│  │  ├── /api/faces         - Face recognition              │  │
│  │  ├── /api/analytics     - Statistics                    │  │
│  │  ├── /api/settings      - Configuration                 │  │
│  │  ├── /ws/detections     - Real-time WebSocket           │  │
│  │  └── /ws/camera/{id}    - Video streaming               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                      BUSINESS LOGIC LAYER                       │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │ Camera Manager │  │  Alert Manager │  │  Face Clusterer │  │
│  │  - Stream RTSP │  │  - Level 1-3   │  │  - HDBSCAN      │  │
│  │  - Record      │  │  - Email/SMS   │  │  - Similarity   │  │
│  │  - Detect      │  │  - WhatsApp    │  │  - Auto-group   │  │
│  └────────────────┘  └────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                         AI ENGINE LAYER                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐ │
│  │ YOLO v8    │  │   CLIP     │  │  FaceNet   │  │ MovieNet │ │
│  │ Person     │  │  Action    │  │   Face     │  │  Video   │ │
│  │ Detection  │  │ Detection  │  │Recognition │  │Understanding│
│  │ (BBox)     │  │(Fighting,  │  │ (Embedding)│  │ (Context)│ │
│  │            │  │Loitering)  │  │            │  │          │ │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘ │
│                    CPU/GPU Auto-Detection                       │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │ SQLite DB      │  │  File Storage  │  │  Video Storage  │  │
│  │  - Cameras     │  │  - Known faces │  │  - Recordings   │  │
│  │  - Detections  │  │  - Unknown     │  │  - Alert clips  │  │
│  │  - Persons     │  │  - Snapshots   │  │  - Segments     │  │
│  │  - Alerts      │  │                │  │                 │  │
│  │  - Time logs   │  │                │  │                 │  │
│  └────────────────┘  └────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                    CAMERA SOURCES                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │IP Camera │  │WiFi Cam  │  │USB Cam   │  │ RTSP Stream  │   │
│  │ (RTSP)   │  │ (HTTP)   │  │ (Device) │  │  (Network)   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                   NOTIFICATION CHANNELS                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────────┐ │
│  │  Email   │  │   SMS    │  │       WhatsApp               │ │
│  │  (SMTP)  │  │ (Twilio) │  │     (PyWhatKit)              │ │
│  │   FREE   │  │  FREE    │  │        FREE                  │ │
│  └──────────┘  └──────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 AI Models

### 1. YOLO v8 (Person Detection)

**Purpose**: Detect persons and create bounding boxes

**Model Variants**:
- **CPU**: YOLOv8n (Nano) - 3MB, 80 FPS on CPU
- **GPU**: YOLOv8m (Medium) - 25MB, 200+ FPS on GPU

**Input**: Video frame (640x640)  
**Output**: Bounding boxes with confidence scores

**Features**:
- Real-time detection
- High accuracy (mAP@50: 37.3%)
- Low latency
- Multi-person tracking

### 2. CLIP (Action Detection)

**Purpose**: Classify actions and behaviors

**Model**: OpenAI CLIP ViT-Base-Patch32

**Actions Detected**:
- Standing, Walking, Running
- Fighting, Falling
- Loitering, Sitting
- Vandalizing, Stealing
- Trespassing
- With weapon
- In distress
- Shouting
- Crowding

**Input**: Cropped person image + text prompts  
**Output**: Action label with confidence score

**Advantages**:
- Zero-shot learning
- No retraining needed
- Multi-language support
- Context understanding

### 3. FaceNet (Face Recognition)

**Purpose**: Person identification

**Model**: InceptionResNetV1 (VGGFace2)

**Process**:
1. **Face Detection**: MTCNN finds faces
2. **Alignment**: Face is aligned and cropped
3. **Embedding**: 512-dimensional vector
4. **Comparison**: Cosine similarity
5. **Threshold**: 0.6 for match

**Features**:
- High accuracy (99.3% LFW)
- Fast inference
- Robust to variations
- Works with masks/glasses

### 4. MovieNet Integration

**Purpose**: Video scene understanding

**Implementation**: Through CLIP's video understanding capabilities

**Features**:
- Temporal context
- Scene analysis
- Event detection
- Behavior patterns

---

## 🔧 Backend Components

### 1. AI Engine (`core/ai_engine.py`)

**Responsibilities**:
- Load and manage AI models
- Perform person detection
- Classify actions
- Extract face embeddings
- Handle CPU/GPU switching

**Key Methods**:
```python
- initialize()              # Load all models
- detect_persons()          # YOLO detection
- detect_action()           # CLIP action classification
- detect_faces()            # FaceNet embedding
- compare_faces()           # Face similarity
- _calculate_zone()         # Distance estimation
```

### 2. Camera Manager (`core/camera_manager.py`)

**Responsibilities**:
- Manage camera streams
- Process video frames
- Continuous recording
- Buffer management

**Key Features**:
- RTSP/HTTP/USB support
- Automatic reconnection
- Frame buffering
- Recording segmentation
- Multi-threading

### 3. Alert Manager (`core/alert_manager.py`)

**Responsibilities**:
- Create alerts
- Send notifications
- Manage alert lifecycle

**Alert Levels**:
- **Level 1**: Email + SMS + WhatsApp
- **Level 2**: Email + SMS
- **Level 3**: Email only

**Notification Services**:
- Gmail SMTP (free)
- Twilio SMS (free trial)
- WhatsApp (pywhatkit)

### 4. Database Layer (`core/database.py`)

**ORM**: SQLAlchemy (async)

**Tables**:
- `cameras` - Camera configurations
- `persons` - Known persons with embeddings
- `detections` - Real-time detections
- `time_logs` - Entry/exit tracking
- `alerts` - Security alerts
- `recordings` - Video metadata
- `unknown_persons` - Unidentified faces
- `system_logs` - Application logs

---

## 🎨 Frontend Components

### 1. Dashboard (`Dashboard.tsx`)

**Features**:
- Multi-view support (Home/About)
- Stats cards
- Camera grid
- Time logs
- Alert viewer

### 2. HomePage (`HomePage.tsx`)

**Advanced Analytics**:
- Time period filters (Today/Week/Month/Year)
- Line charts (Activity timeline)
- Bar charts (Camera performance)
- Pie charts (Person distribution)
- Radar charts (Threat matrix)
- Area charts (Peak hours)
- Live camera feeds

### 3. Camera Grid (`CameraGrid.tsx`)

**Features**:
- 1-16 camera views
- Responsive grid layout
- Real-time detection overlay
- Camera status indicators
- Click to expand

### 4. Person Classifier (`PersonClassifier.tsx`)

**Features**:
- Face image upload
- Person registration
- Classification (Employee/Owner/Visitor)
- Face gallery
- Search and filter

### 5. Alert System (`SuspiciousActivityAlert.tsx`)

**Features**:
- Real-time alerts
- Alert details modal
- Video playback
- Acknowledge/Resolve
- Alert history

### 6. API Client (`services/api-client.ts`)

**Features**:
- Type-safe API calls
- Error handling
- WebSocket management
- Token refresh
- Request interceptors

---

## 🔄 Data Flow

### Detection Pipeline

```
1. Camera Feed
   ↓
2. Frame Capture (30 FPS)
   ↓
3. YOLO Detection
   ├── Person detected?
   │   ↓ YES
   ├── Extract Bounding Box
   ↓
4. CLIP Action Detection
   ├── What action?
   ↓
5. Face Detection (MTCNN)
   ├── Face found?
   │   ↓ YES
   ├── FaceNet Embedding
   ↓
6. Face Matching
   ├── Known person?
   │   ↓ YES → Update last_seen
   │   ↓ NO  → Save to unknown_persons
   ↓
7. Zone Calculation
   ├── Based on bbox size
   ├── Zone 1/2/3
   ↓
8. Alert Evaluation
   ├── Is action suspicious?
   │   ↓ YES
   ├── Determine level (1-3)
   ├── Create alert
   ├── Send notifications
   ↓
9. Save to Database
   ├── Detection record
   ├── Time log update
   ├── Alert record
   ↓
10. WebSocket Broadcast
    ├── Send to frontend
    └── Update UI
```

### Recording Pipeline

```
1. Camera Stream
   ↓
2. Continuous Recording
   ├── 5-minute segments
   ├── MP4 format
   ↓
3. Save to disk
   ├── /data/recordings/
   ↓
4. Database metadata
   ├── file_path
   ├── duration
   ├── size
   ↓
5. Alert clip extraction
   ├── 30 seconds before/after
   ├── Save to /data/alerts/
```

---

## 🗄️ Database Schema

```sql
-- Cameras
CREATE TABLE cameras (
    id INTEGER PRIMARY KEY,
    camera_id TEXT UNIQUE,
    name TEXT,
    location TEXT,
    stream_url TEXT,
    camera_type TEXT,
    status TEXT,
    fps INTEGER,
    resolution TEXT,
    zone_config TEXT,
    created_at TIMESTAMP
);

-- Persons
CREATE TABLE persons (
    id INTEGER PRIMARY KEY,
    person_id TEXT UNIQUE,
    name TEXT,
    classification TEXT,  -- employee/owner/unknown
    face_embedding BLOB,   -- 512D vector
    face_image_path TEXT,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    total_appearances INTEGER,
    notes TEXT
);

-- Detections
CREATE TABLE detections (
    id INTEGER PRIMARY KEY,
    camera_id INTEGER FOREIGN KEY,
    person_id INTEGER FOREIGN KEY,
    bbox_x1 FLOAT,
    bbox_y1 FLOAT,
    bbox_x2 FLOAT,
    bbox_y2 FLOAT,
    confidence FLOAT,
    action TEXT,
    action_confidence FLOAT,
    zone INTEGER,
    distance_from_camera FLOAT,
    timestamp TIMESTAMP,
    frame_path TEXT
);

-- Alerts
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    alert_id TEXT UNIQUE,
    alert_type TEXT,
    alert_level INTEGER,   -- 1/2/3
    severity TEXT,         -- critical/high/medium/low
    camera_id INTEGER FOREIGN KEY,
    zone INTEGER,
    description TEXT,
    action_detected TEXT,
    video_path TEXT,
    snapshot_path TEXT,
    status TEXT,           -- active/acknowledged/resolved
    email_sent BOOLEAN,
    sms_sent BOOLEAN,
    whatsapp_sent BOOLEAN,
    created_at TIMESTAMP,
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Time Logs
CREATE TABLE time_logs (
    id INTEGER PRIMARY KEY,
    person_id INTEGER FOREIGN KEY,
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    duration INTEGER,      -- seconds
    entry_camera TEXT,
    exit_camera TEXT,
    date TIMESTAMP
);

-- Unknown Persons
CREATE TABLE unknown_persons (
    id INTEGER PRIMARY KEY,
    face_embedding BLOB,
    face_image_path TEXT,
    camera_id INTEGER FOREIGN KEY,
    detection_time TIMESTAMP,
    cluster_id INTEGER,
    is_clustered BOOLEAN,
    alert_sent BOOLEAN
);
```

---

## 🔌 API Architecture

### RESTful Endpoints

**Base URL**: `http://localhost:8000`

**Authentication**: Optional (can add JWT)

**Response Format**:
```json
{
  "data": { ... },
  "error": null
}
```

### WebSocket Endpoints

**Real-time Detections**:
```
ws://localhost:8000/ws/detections
```

**Camera Stream**:
```
ws://localhost:8000/ws/camera/{camera_id}
```

---

## 🔒 Security Features

### 1. Face Recognition
- 512D embeddings
- Cosine similarity matching
- Threshold-based authentication
- No image storage (only embeddings)

### 2. Encrypted Storage
- Face embeddings as binary blobs
- Secure password hashing (bcrypt)
- API token encryption

### 3. Network Security
- CORS protection
- Rate limiting
- Input validation
- SQL injection prevention

### 4. Privacy
- Local storage only
- No cloud uploads
- GDPR compliant
- Data retention policies

---

## ⚡ Performance Optimizations

### CPU Mode
- YOLO Nano model (3MB)
- Frame skipping (every 2nd frame)
- Batch size: 1
- Resolution: 640x640
- Single worker thread

### GPU Mode
- YOLO Medium model (25MB)
- Process all frames
- Batch size: 8
- Half precision (FP16)
- Multi-threading
- CUDA streams

### Caching
- Face embedding cache
- Model output cache
- Frame buffer cache

### Async Processing
- Async database queries
- Async file I/O
- Async camera streams
- WebSocket async handlers

---

## 📊 Scalability

### Horizontal Scaling
- Multiple backend instances
- Load balancer
- Shared database
- Distributed file storage

### Vertical Scaling
- More GPU VRAM
- More CPU cores
- SSD storage
- Network bandwidth

### Camera Limits
- **CPU**: 2-4 cameras
- **GPU (RTX 2060)**: 8-12 cameras
- **GPU (RTX 4090)**: 16+ cameras

---

## 🎯 Future Enhancements

1. **Cloud Integration**
   - AWS S3 storage
   - Cloud databases
   - Serverless functions

2. **Advanced AI**
   - Pose estimation
   - Emotion detection
   - Crowd analysis
   - Vehicle detection

3. **Mobile App**
   - iOS/Android
   - Push notifications
   - Remote viewing

4. **Enterprise Features**
   - Multi-tenancy
   - Role-based access
   - Audit logs
   - Compliance reports

---

**Architecture designed by Neelaminds Private Limited**  
*Innovation • Security • Excellence*

🇮🇳 **Made in India** | 🛡️ **Protecting what matters most**
