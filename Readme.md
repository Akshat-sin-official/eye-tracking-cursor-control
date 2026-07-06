<div align="center">
  <h1>eye tracking cursor control (lately - Meyes)</h1>
  
</div>


![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-red)
![PyAutoGUI](https://img.shields.io/badge/PyAutoGUI-Cursor%20Control-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Overview

Meyes is a Python project for controlling the mouse with eye movement. It uses a webcam, facial landmark detection, and screen automation to move the cursor and trigger clicks.

## System Architecture

```mermaid
flowchart LR
    A[Webcam Input] --> B[Frame Capture]
    B --> C[Face Landmark Detection]
    C --> D[Eye Position Extraction]
    D --> E[Cursor Mapping]
    E --> F[Mouse Actions]
    F --> G[Screen / Pointer]
```

## How It Works

1. The webcam captures live frames.
2. MediaPipe detects facial landmarks.
3. Eye position is mapped to screen coordinates.
4. PyAutoGUI moves the cursor and performs clicks.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Notes

- Works best in good lighting.
- Requires a working webcam.
- Best suited for prototyping and assistive interaction experiments.

## Contributing

Contributions are welcome. Open an issue first for larger changes.

<div align="center">
  <p>Thanks for checking out Meyes • Contributions welcome • Star the repo if you like it</p>
</div>
