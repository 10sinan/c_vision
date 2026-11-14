# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Overview

Head-tracked real-time 3D parallax ("window into a 3D room" illusion). A webcam tracks the user's head via MediaPipe; the head position drives an off-axis OpenGL projection so a wireframe room/cube behind the screen appears to have real depth. Code comments and UI strings are in Turkish.

## Commands

- Setup: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt` (a `venv/` already exists in the repo dir)
- Run the full app (webcam): `python main.py`
- Run without a camera (mouse acts as head position): `python main.py --mouse`
- User-specific settings live in `settings.json` (gitignored; [settings.py](settings.py)): screen diagonal + aspect → physical size in cm, webcam gap above the screen top, IPD, focal multiplier. On first run (no file) the settings screen is shown first; `P` reopens it at runtime and `Renderer.set_screen(w_cm, h_cm)` applies the size. Defaults reproduce the old hardcoded 15.6" 16:9 laptop (34.5 × 19.4 cm); `off_axis.py` constants are only fallbacks.
- Distance calibration ([calibration.py](calibration.py)): menu item or `K`. The user sits at a known distance, ~2 s of `ipd_px` samples are collected (median + MAD spread check), and `focal_mult = z_known * ipd_px / ipd_cm / frame_w` is saved to `settings.json`. The math (`focal_mult_from_measurement`, `evaluate`) is OpenGL-free and tested. The tracker now starts *before* the menu so the camera is warm; `TrackerThread.latest_frame()` returns the last mirrored BGR frame + pose (reference only, no copy).
- Menu previews ([thumbnails.py](thumbnails.py)): each model is rendered once off-screen (`Renderer.draw(..., present=False)` + `grab_frame()`, using its saved `.view.json` view) into `.thumbs/<name>-<mtime>-<size>.png` (gitignored); changed model files regenerate and stale thumbs are pruned. `Menu.prepare_previews` runs at startup and reloads whichever model was loaded before (generation replaces the loaded model).
- On startup `main.py` shows a scene picker ([menu.py](menu.py)) listing `*.glb`/`*.gltf`/`*.obj` in the repo dir plus a no-model option; `M` returns to it at runtime. `--model` skips the menu.
- Show a 3D model directly: `python main.py --model scene.glb` (.glb/.gltf with real textures; .obj geometry only, gray). Combine with `--mouse` for no camera. Model view keys (listed in the `main.py` docstring) adjust scale/cover/depth/yaw/pitch; the view is saved on exit to `<model>.view.json` next to the model and reloaded next time. Model files and `*.view.json` are gitignored.
- Staged demos (development history, each runs standalone): `python stage1_demo.py` (tracking), `stage2_demo.py` (normalize + EMA), `stage3_demo.py` (off-axis render; supports `--mouse`)
- Tests (pure-math modules only): `python -m unittest discover -p "test_*.py"`; single test: `python -m unittest test_off_axis.FrustumTest.test_centered_eye_gives_symmetric_frustum`
- No linter or build step is configured. `off_axis.py` is deliberately OpenGL-free so its math can be tested in isolation.

## Architecture

Data flow, one stage per module (the `stageN_demo.py` files mirror the build-up stages 1–3; `main.py` is stage 4 combining them):

1. [head_tracker.py](head_tracker.py) — `HeadTracker` opens the webcam and runs MediaPipe face landmarks. Supports both the legacy `mp.solutions.face_mesh` and the Tasks API (uses `face_landmarker.task` model file in the repo root), choosing at runtime. Returns pose dict `{x_px, y_px, z_cm}`; eye centers are the midpoints of the eye-corner landmarks (33/133, 362/263) — iris landmarks are deliberately not used because they move with gaze. Depth Z comes from inter-pupillary distance (IPD, 6.3 cm assumed) via a pinhole model using `focal_length_px`. Tasks mode runs FaceLandmarker in VIDEO running mode (`detect_for_video`, strictly increasing ms timestamps) and sets `CAP_PROP_BUFFERSIZE=1`. The webcam is capped at 30 fps in hardware (MJPG doesn't help); detection is ~8–10 ms. It imports mediapipe through a hack that blocks the `sounddevice` import (audio backend unused).
2. [tracker_thread.py](tracker_thread.py) — `TrackerThread` runs camera + MediaPipe on a background thread so rendering never blocks on the camera; the main loop reads `latest_measurement()` → `(pose, stamp, seq)` and uses `seq` to detect new measurements. Sets `failed` if the stream dies.
3. [signal_filter.py](signal_filter.py) — `PoseNormalizer` maps pixels to [-1, 1] (y up, matching OpenGL) and clamps Z to 20–160 cm; `OneEuroFilter` (used by `main.py`) is a speed-adaptive low-pass with per-axis params; `EMAFilter` is still used by the stage demos. Measured camera noise: ~0.5 px in X/Y but ~1.2 cm per frame in Z, so Z gets a much lower cutoff (`FILTER_MIN_CUTOFF`/`FILTER_BETA` in `main.py`). The filter must be updated only on new camera measurements (~30 Hz) with their timestamps, not every 60 Hz render frame. Between measurements `main.py` renders `filt.predict(now, PREDICT_HORIZON_S, PREDICT_AXES)` (velocity extrapolation of X/Y only) for smooth 60 Hz motion.
4. [off_axis.py](off_axis.py) — pure math. `head_to_eye_cm` converts normalized pose to eye position in cm relative to screen center (adds webcam vertical offset since the camera sits above the screen); `off_axis_frustum` yields asymmetric `glFrustum` bounds. Coordinate system: screen plane z=0 centered at origin, room extends to z<0, viewer at z>0, X right, Y up, units cm. Screen size constants (34.5 × 19.4 cm, 15.6" 16:9) are only defaults here; the real size comes from `Settings.screen_cm()` and is passed to `off_axis_frustum`.
5. [model_loader.py](model_loader.py) — OpenGL-free loader. glTF goes through trimesh; the scene graph is walked manually and node transforms applied in numpy — do not use `scene.dump()`, it deep-copies textures per mesh and used >3.5 GB on an 85-mesh scene and yields a list of parts: vertices, smooth normals, faces, UVs, PIL texture image, base color, optional vertex colors. trimesh UVs are bottom-left origin, so `renderer._upload_texture` flips the image vertically.
6. [renderer.py](renderer.py) — pygame + PyOpenGL fixed-function pipeline; `draw(eye, dt)` each frame. Without a model: wireframe room (front face = physical screen) + spinning cube. With a model: textures uploaded once (deduplicated by image), all parts compiled into one display list (vertex arrays), drawn lit with alpha test (foliage) and two-sided lighting. `Renderer.load_model(path | None)` can be called repeatedly (menu switching): it saves the previous model's view and frees its display list and textures. Text in menus is drawn with pygame.font → `glWindowPos2d` + `glDrawPixels`. Model is scaled to cover (default) or fit the screen W×H and placed so its front sits `MODEL_GAP_CM` behind the screen plane, then offset/rotated by the view settings. `GL_CLIP_PLANE0` cuts everything in front of the screen plane (z > 0), so enclosed environments can be viewed from inside like through a window.

To check rendering without looking at the screen: monkeypatch `pygame.display.flip` to a no-op, call `draw()`, then `glReadPixels` from `GL_BACK` (reading `GL_FRONT` returns black on this Wayland setup).

[main.py](main.py) wires these together: the same `focal_px` (= `FRAME_W * focal_mult`) must be used for both the Z estimate (set on `tracker.tracker.focal_length_px`) and the X/Y conversion in `head_to_eye_cm`, otherwise depth and lateral scale disagree. When the face is lost for >1 s, the filtered eye drifts back toward center at the neutral 60 cm. Runtime tuning keys (focal multiplier, camera offset, EMA alpha, reset) are documented in the `main.py` docstring.

## Notes

- `face_landmarker.task` (~3.7 MB model) is committed. `.gitignore` excludes `__pycache__/` and `venv/`.
- `mediapipe<1.0` is pinned in requirements.
