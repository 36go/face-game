<div align="center">

# 🎭 Real-Time Expression & Gesture Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10--3.12-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.5+-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-1.0.0-red.svg" alt="MediaPipe">
  <img src="https://img.shields.io/badge/License-Custom-yellow.svg" alt="Custom license">
</p>

<p align="center">
  <strong>A powerful real-time computer vision application that detects facial expressions and hand gestures using AI</strong>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-contact">Contact</a>
</p>

---
</div>

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎭 Facial Expression Detection
- 😐 Neutral
- 😊 Smiling
- 😢 Sad
- 😠 Angry

</td>
<td width="50%">

### 👋 Hand Gesture Recognition
- 👍 Thumbs Up (Animated GIF)
- 🤝 Hand Clasping
- ✋ Open/Closed Hands
- 🎯 Special Gestures

</td>
</tr>
</table>

### 🎵 Interactive Sound Trigger
Open your mouth **3 times rapidly** within 2 seconds to:
- 🔊 Play a sound effect
- 🎬 Trigger a special 4-second animation

---

## 🚀 Installation

### Prerequisites
- 📹 Webcam
- 💻 Python 3.10 through 3.12
- 🖥️ Recommended: Intel i7 11th gen or equivalent
- 💾 Minimum 8GB RAM

### Quick Start

```bash
# Clone the repository
git clone https://github.com/36go/face-game.git
cd face-game/face-game

# Install dependencies
python -m pip install -r requirements.txt

# Run the application
python main.py
```

### 📦 Required Files


| File | Description |
|------|-------------|
| `neutral.png` | Neutral expression image |
| `smile.png` | Smiling expression image |
| `sad.png` | Sad expression image |
| `angry.png` | Angry expression image |
| `jew.png` | Clasped hands image |
| `nien.png` | Special gesture image |
| `thumbs_up.gif` | Thumbs up animation |
| `oi.gif` | Mouth trigger animation |
| `sound.mp3` | Sound effect |

### MediaPipe Models

The app uses the supported MediaPipe Tasks `FaceLandmarker` and `HandLandmarker` APIs. On first run it downloads the versioned official Google model bundles into `%LOCALAPPDATA%\face-game\models` on Windows or `~/.cache/face-game/models` on Linux.

Every cached and downloaded model is checked against its pinned byte length and SHA-256 before MediaPipe can load it. Downloads use temporary files and atomic replacement, so interrupted or modified downloads are never used.

| Model | Cache filename | Official immutable-version URL | SHA-256 |
|------|----------------|--------------------------------|---------|
| Face Landmarker float16 v1 | `face_landmarker-float16-v1.task` | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| Hand Landmarker float16 v1 | `hand_landmarker-float16-v1.task` | `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |

- Set `FACE_GAME_MODEL_DIR` to use a managed or pre-populated model directory.
- Set `FACE_GAME_OFFLINE=1` to prohibit downloads. Missing or invalid files then produce an error containing the required URL, cache path, and SHA-256.
- The models are official MediaPipe artifacts distributed by Google. MediaPipe is licensed under [Apache License 2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE). The model files are cached locally and are not redistributed by this repository.

---

## 🎯 How It Works

### 🇬🇧 English

The application uses the **MediaPipe Tasks API** and **OpenCV** to analyze your webcam feed in real-time:

1. **Face Detection**: The system detects your face and analyzes facial landmarks to determine your expression
2. **Hand Tracking**: Recognizes hand positions and gestures using 21 hand landmarks
3. **Expression Mapping**: Displays corresponding images based on detected expressions and gestures
4. **Interactive Triggers**: Special mouth movements trigger sound effects and animations

**Controls:**
- Press `Q` to quit
- The app opens two windows: **Camera** (live feed) and **Expression** (output display)

**Gestures:**
- **Thumbs Up**: Hold thumb up with fingers folded for 0.5 seconds
- **Clasped Hands**: Bring both hands together
- **Mouth Trigger**: Open mouth 3 times quickly (you'll see debug messages in console)

---

### 🇸🇦 العربية

يستخدم التطبيق **MediaPipe** و **OpenCV** لتحليل فيديو الكاميرا في الوقت الفعلي:

1. **كشف الوجه**: يكتشف النظام وجهك ويحلل معالم الوجه لتحديد تعبيرك
2. **تتبع اليد**: يتعرف على مواضع اليد والإيماءات باستخدام 21 نقطة لمعالم اليد
3. **ربط التعبيرات**: يعرض الصور المقابلة بناءً على التعبيرات والإيماءات المكتشفة
4. **المحفزات التفاعلية**: حركات الفم الخاصة تؤدي إلى تشغيل المؤثرات الصوتية والرسوم المتحركة

**التحكم:**
- اضغط `Q` للخروج
- يفتح التطبيق نافذتين: **الكاميرا** (البث المباشر) و **التعبير** (عرض النتيجة)

**الإيماءات:**
- **الإبهام لأعلى**: ارفع إبهامك مع طي الأصابع لمدة 0.5 ثانية
- **اليدين المتشابكتين**: اجمع كلتا يديك معاً
- **محفز الفم**: افتح فمك 3 مرات بسرعة (سترى رسائل تصحيح في الكونسول)

---
## 🎬 Demo

### Expression Detection
The app displays different images based on your facial expressions in real-time.

### Hand Gestures
- Hold a **thumbs up** gesture to see an animated GIF
- **Clasp your hands** together to trigger a special image
- Other gestures are automatically detected and displayed

### Sound Trigger
Open your mouth **3 times within 2 seconds** to play a sound and show a 4-second animation!

---

## 🛠️ Technical Details

### Performance Optimization
- ⚡ Processes every frame for smooth video
- 🎯 Uses pinned MediaPipe float16 Face and Hand Landmarker models
- 👤 Maximum 1 face detection per frame
- 📐 640x480 resolution for balanced quality and performance

### Dependencies
```txt
opencv-contrib-python==4.14.0.94
mediapipe==1.0.0
numpy==2.2.6
Pillow==12.3.0
protobuf==6.33.5
pygame==2.6.1
```

---

## 📱 Contact

<div align="center">

### 👨‍💻 Made By Me | صُنع بواسطتي
### انا و عبدالله 

<p>
  <strong>تابعوني على حساباتي | Follow Me On My Accounts</strong>
</p>

<p>
  <a href="https://instagram.com/pj.cy">
    <img src="https://img.shields.io/badge/Instagram-@pj.cy-E4405F?style=for-the-badge&logo=instagram&logoColor=white" alt="Instagram pj.cy">
  </a>
  <br>
  <a href="https://instagram.com/bqq7k">
    <img src="https://img.shields.io/badge/Instagram-@bqq7k-E4405F?style=for-the-badge&logo=instagram&logoColor=white" alt="Instagram bqq7k">
  </a>
</p>

<p>
  <a href="https://instagram.com/pj.cy">
    <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" width="40" alt="Instagram">
  </a>
</p>

</div>

---

## 📄 License

This project uses the repository's custom attribution license. See `face-game/LICENSE` for its permissions, conditions, and limitations.

---

## 🙏 Acknowledgments

- **MediaPipe** - Google's ML solution for hand and face detection
- **OpenCV** - Computer vision library
- **PyGame** - Sound playback functionality

---

<div align="center">

### ⭐ Star this repo if you found it helpful!

**Made with ❤️ by**

[![Instagram](https://img.shields.io/badge/@pj.cy-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/pj.cy)
</div>
