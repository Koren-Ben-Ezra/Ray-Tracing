import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

from Ray import Ray
from tqdm import trange
from matplotlib import pyplot as plt

def parse_scene_file(file_path):
    surfaces = []
    materials = []
    lights = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                materials.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                surfaces.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                surfaces.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                surfaces.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                lights.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, surfaces, materials, lights


def save_image(image_array):
    global outputname
    image_array = np.clip(image_array * 255, 0, 255)
    image = Image.fromarray(image_array.astype(np.uint8))

    # Save the image to a file
    image.save(outputname)

def AlignCamera(camera: Camera):
    DirVector = (camera.look_at - camera.position)/np.linalg.norm(camera.look_at - camera.position)
    if np.dot(DirVector, camera.up_vector) != 0: #if direction vector and up vector are not perpendicular
        camera.up_vector = camera.up_vector - np.dot(DirVector, camera.up_vector) * DirVector
        camera.up_vector = camera.up_vector/np.linalg.norm(camera.up_vector)

    return camera

def GetCameraRays(camera: Camera, width: int, height: int):
    forward = camera.look_at - camera.position
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, camera.up_vector)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    screen_center = camera.position + forward * camera.screen_distance

    aspect_ratio = width / float(height)
    half_width = camera.screen_width / 2.0
    half_height = half_width / aspect_ratio

    rays = np.empty((height, width), dtype=object)

    for j in range(height):
        for i in range(width):
            u = ( (i + 0.5) / width  - 0.5 ) * 2.0
            v = ( (j + 0.5) / height - 0.5 ) * 2.0

            pixel_world = screen_center + u * half_width * right - v * half_height * up
            direction = pixel_world - camera.position
            direction /= np.linalg.norm(direction)

            rays[j, i] = Ray(pixel_world, direction)

    return rays


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    global outputname
    outputname = args.output_image

    # Parse the scene file
    camera, scene_settings, surfaces, materials, lights = parse_scene_file(args.scene_file)
    camera = AlignCamera(camera)
    rays: np.ndarray[Ray] = GetCameraRays(camera, args.width, args.height)
    
    image_array = np.zeros((args.height, args.width, 3), dtype=np.float32)

    for i in trange(args.height):
        for j in range(args.width):
            color = rays[i, j].trace_ray(scene_settings, surfaces, lights, materials)
            image_array[i, j] = np.array(color)
    
    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
