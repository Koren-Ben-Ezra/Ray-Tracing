import numpy as np
from surfaces.surface import Surface

EPSILON = 1e-8

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])
AXES = [X_AXIS, Y_AXIS, Z_AXIS]


class Cube(Surface):

    def __init__(self, position, scale, material_index):
        super().__init__(material_index)
        self.position = np.array(position, dtype=float)
        self.scale = float(scale)
        self.material_index = material_index

    def intersect(self, ray):
        half_scale = self.scale * 0.5
        lower = self.position - half_scale
        upper = self.position + half_scale

        t_min = -np.inf
        t_max = np.inf

        for i in range(3):
            if abs(ray.direction[i]) < EPSILON:
                if ray.origin[i] < lower[i] or ray.origin[i] > upper[i]:
                    return None
            else:
                t1 = (lower[i] - ray.origin[i]) / ray.direction[i]
                t2 = (upper[i] - ray.origin[i]) / ray.direction[i]

                t_near = min(t1, t2)
                t_far = max(t1, t2)

                if t_near > t_min:
                    t_min = t_near
                if t_far < t_max:
                    t_max = t_far

                if t_min > t_max:
                    return None

        if t_min < EPSILON and t_max < EPSILON:
            return None

        t_hit = t_min if t_min > EPSILON else t_max
        intersection_point = ray.origin + t_hit * ray.direction
        return intersection_point

    def compute_normal(self, point_on_surface, ray_direction):
        local_point = point_on_surface - self.position
        axis_index = np.argmax(np.abs(local_point))
        sign = np.sign(local_point[axis_index])
        normal = AXES[axis_index] * sign

        return normal
