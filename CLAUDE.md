# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Halloween-themed web application project for kids. The codebase is currently in its initial stages.

## Tech Stack

- **Frontend**: Vanilla HTML, CSS, and JavaScript (no framework)
- **WebRTC API**: `navigator.mediaDevices.getUserMedia()` for webcam access
- **Canvas API**: For future video processing and spooky filters

## Development Commands

Start a local development server (required for webcam access):
```bash
python3 -m http.server 8000
```

Then open http://localhost:8000 in your browser.

**Note**: Webcam access requires either HTTPS or localhost. Opening `index.html` directly via `file://` protocol will not work in modern browsers.

## Project Structure

- `index.html` - Main application page
- `app.js` - Webcam initialization and stream handling
- `style.css` - Styling and layout

## Future Features

The current implementation is a basic webcam mirror. Planned enhancements:
- Canvas-based video filters (grayscale, invert, etc.)
- Spooky effects and overlays
- Face detection/tracking for interactive effects
