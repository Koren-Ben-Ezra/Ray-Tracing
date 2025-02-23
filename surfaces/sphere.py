from surfaces.surface import Surface
from Ray import Ray
import numpy as np

class Sphere(Surface):
    def __init__(self, position, radius, material_index):
        super().__init__(material_index)
        self.position = np.array(position)
        self.radius = radius
        self.material_index = material_index

    def intersect(self, ray: Ray):
        A = np.dot(ray.direction, ray.direction)
        B = 2*np.dot(ray.direction, ray.origin - self.position)
        C = np.dot(ray.origin - self.position, ray.origin - self.position) - self.radius**2
        discriminant = B**2 - 4*A*C

        if discriminant < 0:
            return None
        else:
            t1 = (-B + np.sqrt(discriminant))/(2*A)
            t2 = (-B - np.sqrt(discriminant))/(2*A)
            if t1 < 0:
                return None
            elif t2 < 0:
                return t1 * ray.direction + ray.origin
            else:
                return t2 * ray.direction + ray.origin
    
    def compute_normal(self, point_on_surface, ray_direction):
        return np.array((point_on_surface - self.position)/self.radius)