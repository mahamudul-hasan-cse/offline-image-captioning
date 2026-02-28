# 📸 OfflineCaptioning — On-Device Image Captioning for Android

> Generate natural language descriptions of images **entirely on your Android device** — no internet, no cloud, no privacy concerns.

![Android](https://img.shields.io/badge/Android-3DDC84?style=for-the-badge&logo=android&logoColor=white)
![Kotlin](https://img.shields.io/badge/Kotlin-7F52FF?style=for-the-badge&logo=kotlin&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)

---

## 🎯 What is this?

**OfflineCaptioning** is an Android app that uses a Vision-Language AI model (BLIP) to automatically describe what it sees through your camera — all processed locally on your device.

Point your camera at anything. Get a caption instantly. No data ever leaves your phone.

> 📷 *"a laptop computer sitting on top of a desk"*
> 📷 *"a person sitting on a chair near a window"*

---

## ✨ Features

- 🔒 **Fully offline** — works without any internet connection
- ⚡ **Real-time** — caption generated on camera button press
- 🧠 **BLIP-base model** — state-of-the-art vision-language AI
- 📱 **On-device inference** — powered by ONNX Runtime
- 🔐 **Privacy-first** — no images are sent to any server

---

## 🏗️ How It Works

```
📷 Camera (CameraX)
        ↓
🖼️  Image Preprocessing
        ↓
👁️  Vision Encoder  ──→  Image Features
        ↓
📝  Text Decoder    ──→  Token Generation (Greedy Decoding)
        ↓
💬  Caption Output on Screen
```

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Kotlin |
| Camera | CameraX API |
| AI Model | BLIP-base (Salesforce) |
| Model Format | ONNX |
| Inference Engine | ONNX Runtime for Android v1.20.0 |
| Min SDK | Android 8.0 (API 26) |

---

## ⚙️ Installation

### Prerequisites
- Android Studio (latest)
- Android device with Android 8.0+
- ADB installed
- ~1 GB free storage on device
- Python 3.8+ (for model export only)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/mahamudul-hasan-cse/offline-image-captioning.git
cd offline-image-captioning
```

---

### Step 2 — Export the BLIP model (one-time setup)

The model files are not included in this repo due to their size (~900 MB). Export them once using the provided script.

**Install dependencies:**
```bash
pip install torch transformers onnx onnxruntime optimum
```

**Run the export script:**
```bash
python scripts/export_model.py
```

This will generate the following files inside `models/` folder:
```
models/
├── blip_vision_encoder.onnx
├── blip_vision_encoder.onnx.data
├── blip_text_decoder.onnx
├── blip_text_decoder.onnx.data
├── tokenizer.json
└── vocab.txt
```

> ⏱️ Export takes around 5–10 minutes depending on your machine.

---

### Step 3 — Push models to your Android device

Connect your device via USB with USB Debugging enabled, then run:

```bash
# Create the models directory on device
adb shell mkdir -p /sdcard/Android/data/com.example.offlinecaptioning/files/models

# Push all model files
adb push models/blip_vision_encoder.onnx /sdcard/Android/data/com.example.offlinecaptioning/files/models/
adb push models/blip_vision_encoder.onnx.data /sdcard/Android/data/com.example.offlinecaptioning/files/models/
adb push models/blip_text_decoder.onnx /sdcard/Android/data/com.example.offlinecaptioning/files/models/
adb push models/blip_text_decoder.onnx.data /sdcard/Android/data/com.example.offlinecaptioning/files/models/
adb push models/tokenizer.json /sdcard/Android/data/com.example.offlinecaptioning/files/models/
adb push models/vocab.txt /sdcard/Android/data/com.example.offlinecaptioning/files/models/
```

---

### Step 4 — Build and run

1. Open the project in **Android Studio**
2. Click **Sync Project with Gradle Files**
3. Select your device and click **▶ Run**

---

## 📁 Project Structure

```
OfflineCaptioning/
├── app/
│   └── src/main/
│       ├── java/com/example/offlinecaptioning/
│       │   ├── MainActivity.kt          # Camera UI & entry point
│       │   └── CaptioningViewModel.kt   # BLIP inference pipeline
│       ├── res/                         # Layouts and resources
│       └── AndroidManifest.xml
├── scripts/
│   └── export_model.py                  # BLIP → ONNX export script
├── .gitignore
└── README.md
```

---

## 📊 Performance

| Metric | Result |
|--------|--------|
| Model size (total) | ~900 MB (float32) |
| Quantized size | ~224 MB (INT8, planned) |
| Inference | Fully on-device |
| Internet required | ❌ None |

---

## 🗺️ Roadmap

- [x] BLIP-base export to ONNX format
- [x] CameraX integration
- [x] On-device inference pipeline
- [x] Working offline caption generation
- [ ] INT8 quantization (target: ~224 MB)
- [ ] Inference latency benchmarking
- [ ] Support for multiple languages

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 👤 Author

**Md. Mahamudul Hasan**
GitHub: [@mahamudul-hasan-cse](https://github.com/mahamudul-hasan-cse)

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
