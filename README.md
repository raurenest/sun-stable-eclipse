# sun-stable-eclipse

A Python tool to stabilize solar eclipse videos, keeping the solar disk centered in every frame and compensating for telescope tracking movements.

## Description

Solar eclipse videos recorded through a telescope often exhibit abrupt movements and jitter caused by the automatic tracking system making corrections. This script addresses those movements in two passes:

**Pass 1 — Direct solar disk detection:** each frame is analyzed at full resolution, the solar disk is detected using thresholding and geometric fitting, and the frame is shifted so the sun is centered. Works across all eclipse phases: full disk, partial eclipse, and totality (where the dark lunar disk surrounded by the corona is detected instead).

**Pass 2 — Incremental phase correlation:** applies phase correlation between consecutive already-corrected frames to eliminate high-frequency residual jitter left after pass 1. Correlation always runs against the previous corrected frame, preventing error accumulation.

Any frame edges that fall outside the original image are filled with black.

## Requirements

- Python 3.8 or higher
- OpenCV
- NumPy
- tqdm

```bash
pip install opencv-python numpy tqdm
```

## Usage

```bash
python centrar_sol.py input.mp4 output.mp4
```

The output video has the same resolution and framerate as the input.

To generate a 2x speed version with FFmpeg:

```bash
ffmpeg -i output.mp4 -vf "setpts=PTS/2" -an -r 30 output_x2.mp4
```

## Adjustable parameters

The following constants at the top of the script can be modified:

| Parameter | Default | Description |
|---|---|---|
| `ESCALA_DETECCION` | `1.0` | Fraction of the original resolution used for detection (1.0 = full resolution) |
| `UMBRAL_CONFIANZA` | `0.7` | Geometric confidence threshold for validating the solar arc detection |
| `UMBRAL_RADIO_PARCIAL` | `440` | Maximum radius in pixels to classify a frame as partial phase (scale proportionally if `ESCALA_DETECCION` is changed) |
| `MAX_DESPLAZAMIENTO_CORR` | `10` | Maximum inter-frame displacement in px accepted by the pass 2 correlator |
| `ESCALA_CORRELACION` | `0.5` | Resolution fraction used in the correlation pass |

If `ESCALA_DETECCION` is changed, adjust `UMBRAL_RADIO_PARCIAL` proportionally (base value ~110 at scale 0.25).

## Performance notes

Pass 1 processes each frame at full resolution (`ESCALA_DETECCION = 1.0`), giving the best detection accuracy but at the cost of processing time. For 4K video this can take several minutes. For lower resolutions or when speed matters more, `ESCALA_DETECCION` can be reduced to 0.5 or 0.25 with a moderate loss of precision.

Pass 2 runs at 50% resolution by default (`ESCALA_CORRELACION = 0.5`), which is sufficient for detecting the small residual displacements.

## Known limitations

- Some residual jitter may remain during transitions between partial and totality phases, when the solar arc is very thin.
- The script assumes a dark background (black or very low ambient light). It is not optimized for videos taken with a bright sky background.
- The output codec is `mp4v`. For better compatibility or smaller file size, re-encode the result with FFmpeg:

```bash
ffmpeg -i output.mp4 -vcodec libx264 -crf 23 output_h264.mp4
```

## License

Copyright (C) 2026 Raúl Rengel Estévez

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY. See the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.html) for more details.
