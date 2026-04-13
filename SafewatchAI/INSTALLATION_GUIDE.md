# 📘 SafeWatch AI - Complete Installation Guide

**Smart Security, Smarter Safety**  
*Powered by Neelaminds Private Limited*

---

## 🎯 Table of Contents

1. [System Requirements](#system-requirements)
2. [Backend Installation](#backend-installation)
3. [Frontend Installation](#frontend-installation)
4. [Camera Setup](#camera-setup)
5. [Free Messaging Services](#free-messaging-services)
6. [Running the Application](#running-the-application)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

---

## 💻 System Requirements

### Minimum Requirements (CPU Mode)
- **OS**: Windows 10/11, Linux Ubuntu 20.04+, macOS 11+
- **CPU**: Intel i5 (4 cores) or AMD equivalent
- **RAM**: 8 GB
- **Storage**: 100 GB free space
- **Internet**: Stable connection for model downloads

### Recommended Requirements (GPU Mode)
- **OS**: Windows 10/11, Linux Ubuntu 20.04+
- **CPU**: Intel i7 (8 cores) or AMD equivalent
- **GPU**: NVIDIA RTX 2060 or better (6GB+ VRAM)
- **RAM**: 16 GB or more
- **Storage**: 500 GB+ SSD
- **CUDA**: 11.8 or higher

---

## 🔧 Backend Installation

### Step 1: Install Python

**Windows:**
1. Download Python 3.10 from https://www.python.org/downloads/
2. ✅ Check "Add Python to PATH" during installation
3. Verify: `python --version`

**Linux:**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

**macOS:**
```bash
brew install python@3.10
```

### Step 2: Install FFmpeg

**Windows:**
1. Download from: https://www.gyan.dev/ffmpeg/builds/
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to System PATH
4. Verify: `ffmpeg -version`

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### Step 3: (Optional) Install CUDA for GPU

**Only if you have NVIDIA GPU:**

1. Download CUDA Toolkit 11.8 from: https://developer.nvidia.com/cuda-downloads
2. Install CUDA Toolkit
3. Install cuDNN: https://developer.nvidia.com/cudnn
4. Verify: `nvidia-smi`

### Step 4: Clone Repository

```bash
git clone https://github.com/your-org/safewatch-ai.git
cd safewatch-ai
```

### Step 5: Setup Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**⏱️ Note**: Installation may take 10-30 minutes depending on your internet speed

### Step 6: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your credentials
# Windows: notepad .env
# Linux/Mac: nano .env
```

### Step 7: Download AI Models (Automatic)

The models will be downloaded automatically on first run:
- YOLO v8 Nano (~6 MB)
- CLIP ViT Base (~350 MB)
- FaceNet (~90 MB)

**Total Download**: ~450 MB

---

## 🎨 Frontend Installation

### Step 1: Install Node.js

Download and install Node.js 18+ from: https://nodejs.org/

Verify installation:
```bash
node --version
npm --version
```

### Step 2: Setup Frontend

```bash
# From project root
cd safewatch-ai

# Install dependencies
npm install

# Or using pnpm (faster)
npm install -g pnpm
pnpm install
```

### Step 3: Configure API URL

Create `.env` file in project root:
```bash
VITE_API_URL=http://localhost:8000
```

---

## 📹 Camera Setup

### IP Camera (RTSP)

**Find your camera's RTSP URL:**

Common formats:
```
# Generic
rtsp://username:password@192.168.1.100:554/stream

# Hikvision
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101

# Dahua
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0

# Axis
rtsp://root:password@192.168.1.100/axis-media/media.amp

# Foscam
rtsp://username:password@192.168.1.100:554/videoMain

# Amcrest
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
```

**Test RTSP with VLC:**
1. Open VLC Media Player
2. Media → Open Network Stream
3. Enter your RTSP URL
4. Click Play

### WiFi Camera (HTTP/MJPEG)

```
# Generic
http://192.168.1.100:8080/video

# ESP32-CAM
http://192.168.1.100:81/stream

# DroidCam
http://192.168.1.100:4747/video
```

### USB Camera

```
# Windows
0  # First camera
1  # Second camera

# Linux
/dev/video0
/dev/video1
```

### Add Camera via API

```bash
curl -X POST "http://localhost:8000/api/cameras/add" \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "cam-001",
    "name": "Main Entrance",
    "location": "Front Door",
    "stream_url": "rtsp://admin:password@192.168.1.100:554/stream",
    "camera_type": "IP",
    "fps": 30,
    "resolution": "1920x1080"
  }'
```

---

## 📧 Free Messaging Services

### 1. Email Setup (Gmail - FREE)

**Step 1**: Enable 2-Factor Authentication
1. Go to: https://myaccount.google.com/security
2. Enable "2-Step Verification"

**Step 2**: Generate App Password
1. Go to: https://myaccount.google.com/apppasswords
2. Select "App": Mail
3. Select "Device": Other (Custom name) → "SafeWatch AI"
4. Click "Generate"
5. Copy the 16-digit password

**Step 3**: Add to `.env`
```env
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=abcd efgh ijkl mnop  # 16-digit app password
ALERT_EMAIL_RECIPIENTS=["security@company.com"]
```

### 2. SMS Setup (Twilio - FREE $15 Credit)

**Step 1**: Create Account
1. Go to: https://www.twilio.com/try-twilio
2. Sign up for free trial
3. Verify your phone number
4. Get $15 free credit

**Step 2**: Get Credentials
1. Dashboard → Account Info
2. Copy:
   - Account SID
   - Auth Token
3. Get a Twilio phone number (free trial)

**Step 3**: Add to `.env`
```env
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890
ALERT_SMS_RECIPIENTS=["+1234567890"]
```

**Limitations**: 
- Free trial: Requires phone number verification
- Can only send to verified numbers
- $15 credit (~500 SMS)

### 3. WhatsApp Setup (FREE - No API Key)

**Step 1**: Install WhatsApp on your computer  
**Step 2**: Keep WhatsApp Web logged in  
**Step 3**: Add to `.env`

```env
WHATSAPP_RECIPIENTS=["+1234567890"]
```

**Note**: 
- Uses WhatsApp Web automation
- Computer must be running
- WhatsApp must be logged in

**Alternative**: Use WhatsApp Business API (requires approval)

---

## 🚀 Running the Application

### Start Backend Server

```bash
cd backend

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Start server
python main.py
```

Expected output:
```
========================================
SafeWatch AI - System Configuration
========================================
🖥️  Device: CPU
🖥️  Running on CPU - Optimized for performance
⚙️  Batch Size: 1
📦 YOLO Model: yolov8n.pt
🤖 AI Models: YOLO + CLIP + FaceNet + MovieNet
========================================

🚀 Starting SafeWatch AI Backend...
✅ Database initialized
✅ AI Models loaded
✅ Camera Manager ready
✅ Alert Manager initialized
🎉 SafeWatch AI Backend is ready!

INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Start Frontend

**Open a new terminal:**

```bash
cd safewatch-ai

# Development mode
npm run dev

# Or with pnpm
pnpm dev
```

Expected output:
```
  VITE v6.3.5  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.1.100:5173/
```

### Access Application

Open browser: **http://localhost:5173/**

**Default Login** (if authentication is enabled):
- Username: `admin`
- Password: `admin123`

---

## 🧪 Testing

### 1. Test Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "ai_engine": {
    "ready": true,
    "device": "cpu",
    "models": {
      "yolo": true,
      "clip": true,
      "face_detector": true,
      "face_recognizer": true
    }
  },
  "active_cameras": 0,
  "database": "connected"
}
```

### 2. Test API Documentation

Visit: **http://localhost:8000/docs**

Interactive Swagger UI with all API endpoints

### 3. Test Camera Feed

Add a test camera and check if stream works

### 4. Test Detections

Use a test video or webcam to verify person detection

---

## 🐛 Troubleshooting

### Backend Issues

**Issue**: `ModuleNotFoundError: No module named 'torch'`  
**Solution**: 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Issue**: `CUDA not available`  
**Solution**: Install CUDA Toolkit or run in CPU mode (automatic)

**Issue**: `FFmpeg not found`  
**Solution**: Install FFmpeg and add to PATH

**Issue**: `Port 8000 already in use`  
**Solution**: Change port in `.env`: `PORT=8001`

### Frontend Issues

**Issue**: `Cannot connect to backend`  
**Solution**: 
1. Check backend is running
2. Verify `VITE_API_URL` in `.env`
3. Check for CORS errors in browser console

**Issue**: `npm install fails`  
**Solution**: 
```bash
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### Camera Issues

**Issue**: `Camera stream not working`  
**Solution**:
1. Test RTSP URL in VLC
2. Check camera is on same network
3. Verify username/password
4. Check firewall settings

**Issue**: `Low FPS / Lag`  
**Solution**:
1. Reduce camera resolution
2. Lower FPS setting
3. Use GPU if available
4. Reduce number of active cameras

### Performance Issues

**Issue**: `High CPU usage`  
**Solution**:
1. Use YOLO Nano model
2. Increase frame skip interval
3. Reduce camera resolution
4. Disable face recognition temporarily

**Issue**: `Out of memory`  
**Solution**:
1. Reduce batch size
2. Use smaller YOLO model
3. Limit number of cameras
4. Add more RAM

---

## 📊 Performance Benchmarks

### CPU Mode (Intel i5)
- **Cameras**: 2-4 simultaneous
- **FPS**: 5-10 per camera
- **Latency**: 200-500ms
- **RAM**: 4-6 GB

### GPU Mode (RTX 3060)
- **Cameras**: 8-16 simultaneous
- **FPS**: 20-30 per camera
- **Latency**: 50-100ms
- **RAM**: 8-12 GB
- **VRAM**: 4-6 GB

---

## 📞 Support

**Neelaminds Private Limited**

🌐 **Website**: https://www.neelaminds.com  
📧 **Email**: support@neelaminds.com  
📱 **Phone**: +91-XXXX-XXXXXX  
🇮🇳 **Made in India with** ❤️

---

## ✅ Quick Start Checklist

- [ ] Python 3.10+ installed
- [ ] Node.js 18+ installed
- [ ] FFmpeg installed
- [ ] Repository cloned
- [ ] Backend dependencies installed
- [ ] Frontend dependencies installed
- [ ] `.env` files configured
- [ ] Email/SMS credentials added
- [ ] Backend server running
- [ ] Frontend dev server running
- [ ] First camera added
- [ ] First detection successful

**Congratulations!** 🎉 SafeWatch AI is now running!

---

## 🎓 Next Steps

1. Add your cameras
2. Register known persons
3. Configure alert levels
4. Set up zone boundaries
5. Test emergency scenarios
6. Review documentation
7. Join support community

**SafeWatch AI** - Protecting what matters most! 🛡️
