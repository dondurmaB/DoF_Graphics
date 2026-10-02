# Experiment 13 — Physical Focus Distance and Circle of Confusion

## Goal

Calculate a physically motivated Circle of Confusion (CoC) from the sampled and linearized depth texture. This experiment visualizes how much blur each pixel should receive later, without applying any blur yet.

## Pipeline

Scene Depth Texture
→ Raw Depth
→ Linear View Depth (meters)
→ Thin-Lens CoC
→ Sensor-Space CoC
→ Pixel-Space CoC
→ Debug Visualization

## What Changed

- Adopted the scene convention `1 world unit = 1 meter`.
- Added focus distance, focal length, f-number, sensor height, and CoC display controls.
- Extended `ScreenMode` with `CoCMagnitude` and `CoCSigned`.
- Calculated signed sensor-space CoC in the screen shader.
- Converted sensor-space CoC to approximate pixel radius using framebuffer height.
- Added grayscale CoC magnitude and red/blue signed CoC visualizations.

## Key Concepts

### Physical Scene Units

The project now treats `1 world unit = 1 meter`. Reconstructed linear view depth is interpreted as meters.

### Focus Distance

`focusDistanceMeters` is measured from the camera along the view-depth direction. Geometry near this depth has CoC near zero.

### Focal Length

`focalLengthMillimeters` is converted from millimeters to meters in the shader. In this experiment it affects only the CoC model, not the raster camera FOV.

### F-number

`fNumber` controls aperture size. Lower f-numbers produce a larger aperture and stronger defocus; higher f-numbers reduce CoC magnitude.

### Aperture Diameter

Aperture diameter is computed as focal length divided by f-number.

### Circle of Confusion

CoC estimates how large a point projects on the sensor when it is not at the focus distance. Larger absolute CoC means stronger potential blur.

### Signed CoC

This implementation uses negative CoC for foreground/near defocus and positive CoC for background/far defocus. Signed mode visualizes foreground as red and background as blue.

### CoC in Pixels

Sensor-space CoC is converted to pixels using sensor height and framebuffer height. This approximate pixel radius will later become a blur-radius input.

### Focus Plane

The focus plane is the depth where `z = z_f`, producing CoC near zero.

## Important Formula

Aperture diameter:

```text
A = f / N
```

Signed CoC:

```text
c = [A f (z - z_f)] / [z (z_f - f)]
```

Definitions:

- `z` = object/view depth in meters
- `z_f` = focus distance in meters
- `f` = focal length in meters
- `A` = aperture diameter in meters
- `N` = f-number

Pixel conversion:

```text
CoC_pixels = (CoC_sensor / sensor_height) * framebuffer_height
```

## Important Shader Values

| Item | Purpose |
|---|---|
| `uFocusDistanceMeters` | Depth from the camera that should be in focus. |
| `uFocalLengthMillimeters` | Physical focal length used by the CoC model. |
| `uFNumber` | Aperture control; lower values increase CoC. |
| `uSensorHeightMillimeters` | Converts sensor-space CoC to pixel-space CoC. |
| `uFramebufferHeightPixels` | Current framebuffer height for pixel conversion. |
| `uCoCVisualizationMaxPixels` | Display-only scale for CoC debug visualization. |

## Experiment Controls

| Control | Effect | Try |
|---|---|---|
| `screenMode` | Chooses color/depth/CoC visualization | `Color`, `CoCMagnitude`, `CoCSigned` |
| `focusDistanceMeters` | Moves the sharp depth region | `3.0f`, `5.0f`, `10.0f` |
| `focalLengthMillimeters` | Changes CoC strength | `35.0f`, `50.0f`, `85.0f` |
| `fNumber` | Changes aperture diameter | `1.4f`, `2.8f`, `8.0f`, `16.0f` |
| `sensorHeightMillimeters` | Changes sensor-to-pixel CoC conversion | `18.0f`, `24.0f`, `36.0f` |
| `cocVisualizationMaxPixels` | Display-only normalization for CoC | `5.0f`, `20.0f`, `40.0f` |

## Observations

- CoC magnitude is dark near the focus distance.
- Nearer and farther geometry becomes brighter as defocus increases.
- Signed mode separates foreground and background defocus.
- Lower f-number values increase CoC magnitude.
- Larger focal lengths generally increase CoC magnitude.
- Changing `cocVisualizationMaxPixels` only changes debug display scaling.

## Remember

- This experiment calculates blur amount only; it does not blur the image.
- The current raster camera still uses `fieldOfViewDegrees`.
- Focal length and sensor height are used for the CoC model only.
- A later refinement may derive FOV consistently from focal length and sensor size.
