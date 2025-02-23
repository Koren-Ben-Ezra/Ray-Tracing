from surfaces.surface import Surface
import numpy as np

class InfinitePlane(Surface):
    def __init__(self, normal, offset, material_index):
        super().__init__(material_index)
        self.normal = np.array(normal)
        norm = np.linalg.norm(self.normal)
        self.offset = offset
        if norm != 0:
            self.offset /= norm
            self.normal /= norm
        else:
            self.normal = np.array([1, 0, 0])
            
        
        self.material_index = material_index

    def intersect(self, ray):
        DotProd = self.normal @ ray.direction
        if np.abs(DotProd) < 1e-12:
            return None

        t = (self.offset - self.normal @ ray.origin) / DotProd
        if t < 0:
            return None

        return ray.origin + t*ray.direction
    
    def compute_normal(self, point_on_surface, ray_direction):
        DotProd = np.dot(self.normal, ray_direction)
        if DotProd > 0:
            return -self.normal
        else:
            return self.normal