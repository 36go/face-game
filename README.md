<div align="center">

# 🎭 Real-Time Expression & Gesture Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.5+-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-Latest-red.svg" alt="MediaPipe">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
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
- 💻 Python 3.8 or higher
- 🖥️ Recommended: Intel i7 11th gen or equivalent
- 💾 Minimum 8GB RAM

### Quick Start

```bash
# Clone the repository
git clone https://github.com/36go/face-game.git
cd face-game

# Install dependencies
pip install -r requirements.txt

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

---

## 🎯 How It Works

### 🇬🇧 English

The application uses **MediaPipe** and **OpenCV** to analyze your webcam feed in real-time:

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
- 🎯 Uses MediaPipe's lightweight model (complexity=0)
- 👤 Maximum 1 face detection per frame
- 📐 640x480 resolution for balanced quality and performance

### Dependencies
```txt
opencv-python
mediapipe
pillow
numpy
pygame
```

---

## 📱 Contact

<div align="center">

### 👨‍💻 Made By Me | صُنع بواسطتي

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
  <a href="https://instagram.com/qqsju">
    <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" width="40" alt="Instagram">
  </a>
</p>

</div>

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **MediaPipe** - Google's ML solution for hand and face detection
- **OpenCV** - Computer vision library
- **PyGame** - Sound playback functionality

---

<div align="center">

### ⭐ Star this repo if you found it helpful!

**Made with ❤️ by**

[![Instagram](https://img.shields.io/badge/@pj.cy-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/[pj.cy])
[![Instagram](https://img.shields.io/badge/@bqq7k-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/bqq7k)
</div>
