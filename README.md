# Ray Tracing Renderer

## Overview
Basic ray tracer implemented in Python as part of a Computer Graphics assignment. The ray tracer simulates light interactions by casting rays from a virtual camera into a 3D scene, determining intersections with objects, and calculating pixel colors based on materials and lighting.

## Teaser
![image](https://github.com/user-attachments/assets/69aee09f-6052-45b1-bf6d-47cfbd009f0d)



## Features
- **Ray-Object Intersections**: Supports spheres, planes, and cubes.
- **Material Properties**: Implements diffuse, specular, reflective, and transparent surfaces using the Phong shading model.
- **Lighting and Shadows**: Uses point lights with soft shadows for realistic shading.
- **Recursive Ray Tracing**: Handles reflection and refraction for enhanced realism.
- **Custom Scene Files**: Parses scene descriptions from simple text files.

## Installation & Usage

### Requirements
Ensure Python 3.x is installed along with the required dependencies:
```bash
pip install numpy pillow
```

### Running the Ray Tracer
Execute the ray tracer with a scene file:
```bash
python raytracer.py scenes/sample_scene.txt output.png --width 500 --height 500
```
- `sample_scene.txt`: Scene file defining objects, lights, and camera.
- `output.png`: Output image file.
- `--width`, `--height`: Optional parameters for image resolution (default: `500x500`).

### Scene File Format
Scenes are described using a simple text format. Example:
```
cam 0 0 0  0 0 -1  0 1 0  1.5 2
set 0 0 0  5 10
mtl 1 0 0  1 1 1  0.5 0.5 0.5  50 0
sph 0 0 -5 1 1
lgt 2 2 -3  1 1 1  1 1 0.5
```
- Defines camera, background color, materials, spheres, and lights.
- Uses simple numerical parameters for object properties.

## Implementation Details
1. **Ray Generation**: Computes primary rays from the camera through each pixel.
2. **Intersection Tests**: Determines nearest surface intersections.
3. **Shading Computation**:
   - Diffuse and specular reflection using the Phong model.
   - Soft shadows using multiple shadow rays.
4. **Recursive Reflection & Transparency**: Handles mirror-like and glass-like materials.
5. **Image Output**: Saves the rendered scene using PIL (Pillow).
