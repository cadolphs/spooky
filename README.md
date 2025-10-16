# spooky

A Halloween-themed webcam mirror web app with spooky real-time filters and effects!

## What it does

Transform yourself with real-time spooky filters:

- **Ghost Head** - Ethereal ghost with displaced colored edges and rainbow eye trails using face detection
- **Haunted** - Inverted grayscale with jittery glowing edges and color-shifting effects
- **Zombify** - Undead transformation with sickly green skin tones and pulsing decay effects
- **Spooky** - Classic inverted grayscale for a simple ghostly look
- **Plus**: Grayscale, Invert, and Face Debug modes

All filters run with real-time face detection powered by TensorFlow.js and MediaPipe FaceMesh!

## Privacy First

**All processing happens 100% client-side in your browser.** Your webcam video never leaves your device - no data is sent to any server. The app works entirely offline once loaded!

## Quick Start

1. Start a local server (required for webcam access):
```bash
python3 -m http.server 8000
```

2. Open http://localhost:8000 in your browser

3. Click "Start Camera" and allow webcam access

4. Choose a filter and get spooked!

## Tech Stack

- Vanilla HTML/CSS/JavaScript
- WebRTC API for webcam access
- TensorFlow.js + MediaPipe FaceMesh for face detection
- Canvas API for real-time image processing
- No server-side code, no build tools needed

## Browser Requirements

Any modern browser (Chrome, Firefox, Safari, Edge) with:
- Webcam support
- JavaScript enabled
- Canvas API support
