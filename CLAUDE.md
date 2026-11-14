# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Overview

Head-tracked real-time 3D parallax ("window into a 3D room" illusion). A webcam tracks the user's head via MediaPipe; the head position drives an off-axis OpenGL projection so a wireframe room/cube behind the screen appears to have real depth. Code comments and UI strings are in Turkish.

## Commands

- Setup: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt` (a `venv/` already exists in the repo dir)
- Run the full app (webcam): `python main.py`
- Run without a camera (mouse acts as head position): `python main.py --mouse`
- Staged demos (development history, each runs standalone): `python stage1_demo.py` (tracking), `stage2_demo.py` (normalize + EMA), `stage3_demo.py` (off-axis render; supports `--mouse`)
- Tests (only `off_axis.py` math is covered): `python -m unittest test_off_axis -v`; single test: `python -m unittest test_off_axis.FrustumTest.test_centered_eye_gives_symmetric_frustum`
- No linter or build step is configured. `off_axis.py` is deliberately OpenGL-free so its math can be tested in isolation.

## Architecture

Data flow, one stage per module (the `stageN_demo.py` files mirror the build-up stages 1–3; `main.py` is stage 4 combining them):

1. [head_tracker.py](head_tracker.py) — `HeadTracker` opens the webcam and runs MediaPipe face landmarks. Supports both the legacy `mp.solutions.face_mesh` and the Tasks API (uses `face_landmarker.task` model file in the repo root), choosing at runtime. Returns pose dict `{x_px, y_px, z_cm}`; depth Z comes from inter-pupillary distance (IPD, 6.3 cm assumed) via a pinhole model using `focal_length_px`. It imports mediapipe through a hack that blocks the `sounddevice` import (audio backend unused).
2. [tracker_thread.py](tracker_thread.py) — `TrackerThread` runs camera + MediaPipe on a background thread so rendering never blocks on the camera; the main loop only reads `latest()` → `(pose, age_seconds)`. Sets `failed` if the stream dies.
3. [signal_filter.py](signal_filter.py) — `PoseNormalizer` maps pixels to [-1, 1] (y up, matching OpenGL) and clamps Z to 20–160 cm; `EMAFilter` smooths the (x, y, z) vector.
4. [off_axis.py](off_axis.py) — pure math. `head_to_eye_cm` converts normalized pose to eye position in cm relative to screen center (adds webcam vertical offset since the camera sits above the screen); `off_axis_frustum` yields asymmetric `glFrustum` bounds. Coordinate system: screen plane z=0 centered at origin, room extends to z<0, viewer at z>0, X right, Y up, units cm. Screen size constants (34.5 × 19.4 cm, 15.6" 16:9) are hardcoded here.
5. [renderer.py](renderer.py) — pygame + PyOpenGL (fixed-function pipeline) wireframe room whose front face coincides with the physical screen; takes eye position each frame in `draw(eye, dt)`.

[main.py](main.py) wires these together: the same `focal_px` (= `FRAME_W * focal_mult`) must be used for both the Z estimate (set on `tracker.tracker.focal_length_px`) and the X/Y conversion in `head_to_eye_cm`, otherwise depth and lateral scale disagree. When the face is lost for >1 s, the filtered eye drifts back toward center at the neutral 60 cm. Runtime tuning keys (focal multiplier, camera offset, EMA alpha, reset) are documented in the `main.py` docstring.

## Notes

- `face_landmarker.task` (~3.7 MB model) is committed. `.gitignore` excludes `__pycache__/` and `venv/`.
- `mediapipe<1.0` is pinned in requirements.
