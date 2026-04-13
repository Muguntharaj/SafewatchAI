# SafeWatch AI - Backend Server

**Smart Security, Smarter Safety**  
*Powered by Neelaminds Private Limited*

---

## 🚀 Features

### AI Models
- ✅ **YOLO v8** - Person detection with bounding boxes
- ✅ **CLIP** - Advanced action detection
- ✅ **FaceNet** - Face recognition and identification
- ✅ **MovieNet** - Video understanding (integrated via CLIP)

### Core Capabilities
- 🎥 **Multi-Camera Support** - IP/WiFi/USB cameras (RTSP, HTTP, RTMP)
- 👤 **Face Recognition** - Automatic person classification (employee/owner/unknown)
- 🎬 **Action Detection** - Fighting, loitering, trespassing, theft, weapons, etc.
- 📍 **3-Zone Detection** - Distance-based zoning from camera
- 🔴 **Continuous Recording** - Automatic session-based recording
- ⚠️ **Smart Alerts** - 3-level alert system with automatic notifications
- 📧 **Multi-Channel Notifications** - Email, SMS, WhatsApp (Free APIs)
- 🗄️ **Local Database** - SQLite for all data storage
- 🖼️ **Face Clustering** - Automatic grouping of unknown persons
- ⚡ **CPU/GPU Compatible** - Optimized for both low-end and high-end systems

---

## 📋 Requirements

### System Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS
- **CPU**: Intel i5 or better (for CPU mode)
- **GPU**: NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 100GB+ free space for recordings

### Software Requirements
- Python 3.8+
- FFmpeg
- CUDA Toolkit (optional, for GPU)

---

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-org/safewatch-ai.git
cd safewatch-ai/backend
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg
**Windows:**
```bash
# Download from https://ffmpeg.org/download.html
# Add to PATH
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 5. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your credentials
```

---

## ⚙️ Configuration

### `.env` File

```env
# Email Configuration (Free - Gmail SMTP)
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
# Note: Use App Password, not regular password
# Generate at: https://myaccount.google.com/apppasswords

# SMS Configuration (Twilio - Free Trial)
TWILIO_ACCOUNT_SID=your-account-sid
TWILIO_AUTH_TOKEN=your-auth-token
TWILIO_PHONE_NUMBER=+1234567890
# Get free trial at: https://www.twilio.com/try-twilio

# Alert Recipients
ALERT_EMAIL_RECIPIENTS=["security@company.com", "admin@company.com"]
ALERT_SMS_RECIPIENTS=["+1234567890", "+0987654321"]
WHATSAPP_RECIPIENTS=["+1234567890"]
```

### Camera Configuration

Add cameras via API or database:

```python
# IP Camera (RTSP)
stream_url = "rtsp://username:password@192.168.1.100:554/stream"

# WiFi Camera (HTTP)
stream_url = "http://192.168.1.101:8080/video"

# USB Camera
stream_url = "0"  # Device index
```

---

## 🚀 Running the Server

### Start Backend Server
```bash
python main.py
```

The server will start on `http://localhost:8000`

### Check Health
```bash
curl http://localhost:8000/health
```

### API Documentation
Visit: `http://localhost:8000/docs`

---

## 📊 API Endpoints

### Cameras
- `POST /api/cameras/add` - Add new camera
- `GET /api/cameras/list` - List all cameras
- `GET /api/cameras/{camera_id}` - Get camera details
- `POST /api/cameras/{camera_id}/start` - Start camera
- `POST /api/cameras/{camera_id}/stop` - Stop camera
- `DELETE /api/cameras/{camera_id}` - Delete camera

### Detections
- `GET /api/detections/recent` - Recent detections
- `GET /api/detections/stats` - Detection statistics
- `GET /api/detections/by-camera/{camera_id}` - Camera-specific detections

### Alerts
- `GET /api/alerts/list` - List alerts
- `GET /api/alerts/{alert_id}` - Get alert details
- `POST /api/alerts/{alert_id}/acknowledge` - Acknowledge alert
- `POST /api/alerts/{alert_id}/resolve` - Resolve alert
- `GET /api/alerts/stats/summary` - Alert statistics

### Face Recognition
- `POST /api/faces/register` - Register new person
- `GET /api/faces/list` - List registered persons
- `GET /api/faces/unknown/list` - List unknown persons
- `POST /api/faces/cluster-unknown` - Cluster unknown faces
- `POST /api/faces/unknown/{id}/classify` - Classify unknown person

### Analytics
- `GET /api/analytics/dashboard` - Dashboard analytics
- `GET /api/analytics/timeline` - Timeline data
- `GET /api/analytics/activity-heatmap` - Activity heatmap

### Settings
- `GET /api/settings/system` - System settings
- `GET /api/settings/notification-config` - Notification config
- `GET /api/settings/storage-info` - Storage information

### WebSocket Endpoints
- `ws://localhost:8000/ws/detections` - Real-time detections
- `ws://localhost:8000/ws/camera/{camera_id}` - Live camera stream

---

## 🎯 Alert Levels

### Level 1 - Critical (Very Harmful)
**Actions:** Fighting, Weapon detected, Violence, Fire, Person falling  
**Notifications:** Email + SMS + WhatsApp  
**Response:** Immediate attention required

### Level 2 - Medium
**Actions:** Running, Shouting, Vandalism, Theft, Trespassing  
**Notifications:** Email + SMS  
**Response:** Quick review needed

### Level 3 - Normal/Mixed
**Actions:** Loitering, Crowding, Suspicious behavior  
**Notifications:** Email only  
**Response:** Monitor situation

---

## 📍 Zone Detection

### Zone 1 (Close)
Distance: 0-2 meters from camera  
Detection: "fighting near camera", "person close to camera"

### Zone 2 (Medium)
Distance: 2-5 meters from camera  
Detection: "person in middle distance"

### Zone 3 (Far)
Distance: 5-10 meters from camera  
Detection: "person far from camera"

---

## 💾 Database Schema

### Tables
- `cameras` - Camera configurations
- `persons` - Registered persons with face embeddings
- `detections` - Real-time person detections
- `time_logs` - Entry/exit time tracking
- `alerts` - Security alerts
- `recordings` - Continuous recording metadata
- `unknown_persons` - Unknown faces for clustering
- `system_logs` - System activity logs

---

## 🔧 Performance Optimization

### CPU Mode (Low-end Systems)
- Uses YOLO Nano model
- Process every 2nd frame
- Batch size: 1
- Single worker thread

### GPU Mode (High-end Systems)
- Uses YOLO Medium/Large model
- Process every frame
- Batch size: 8
- Half precision (FP16)
- Multiple worker threads

**Auto-detection:** System automatically detects and optimizes for CPU/GPU

---

## 📁 Directory Structure

```
backend/
├── main.py                 # Main application
├── requirements.txt        # Dependencies
├── .env                    # Environment variables
├── core/
│   ├── config.py          # Configuration
│   ├── database.py        # Database models
│   ├── ai_engine.py       # AI models (YOLO, CLIP, FaceNet)
│   ├── camera_manager.py  # Camera handling
│   └── alert_manager.py   # Alert system
├── api/
│   ├── cameras.py         # Camera endpoints
│   ├── detections.py      # Detection endpoints
│   ├── alerts.py          # Alert endpoints
│   ├── faces.py           # Face recognition endpoints
│   ├── analytics.py       # Analytics endpoints
│   └── settings.py        # Settings endpoints
└── data/
    ├── recordings/        # Video recordings
    ├── alerts/           # Alert snapshots/videos
    ├── faces/
    │   ├── known/        # Known person faces
    │   └── unknown/      # Unknown person faces
    └── logs/             # System logs
```

---

## 🆓 Free Messaging Services Setup

### Email (Gmail SMTP - Free)
1. Enable 2FA on Gmail
2. Generate App Password: https://myaccount.google.com/apppasswords
3. Use App Password in `.env`

### SMS (Twilio - Free Trial)
1. Sign up at: https://www.twilio.com/try-twilio
2. Get $15 free credit
3. Add credentials to `.env`

### WhatsApp (pywhatkit - Free)
- No API key needed
- Uses WhatsApp Web
- Must keep browser open during send

---

## 🐛 Troubleshooting

### CUDA Not Found
```bash
# Verify CUDA installation
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Camera Connection Issues
- Verify RTSP URL is correct
- Check network connectivity
- Ensure camera supports RTSP/HTTP streaming
- Test with VLC media player first

### Low Performance
- Reduce camera resolution
- Increase frame skip interval
- Use YOLO Nano model
- Disable face recognition for some cameras

---

## 📞 Support

**Neelaminds Private Limited**  
🌐 Website: https://www.neelaminds.com  
📧 Email: support@neelaminds.com  
🇮🇳 Made in India with ❤️

---

## 📄 License

Copyright © 2026 Neelaminds Private Limited. All rights reserved.

---

## 🙏 Acknowledgments

- YOLO by Ultralytics
- CLIP by OpenAI
- FaceNet by Google
- FastAPI Framework
- SQLAlchemy
- OpenCV

**SafeWatch AI** - Protecting what matters most! 🛡️
