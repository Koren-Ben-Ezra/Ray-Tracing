import numpy as np

from material import Material
from light import Light
from scene_settings import SceneSettings
from surfaces.surface import Surface

RGB = tuple[float,float,float]
XYZ = np.ndarray                

class Ray:
    
    max_recursions: int = -1
    background_color: RGB = (128, 128, 0)
    
    materials: list[Material] = []
    lights: list[Light] = []
    surfaces: list[Surface] = []
    
    def __init__(self, origin, direction, rec_iter=0):
        self.origin = origin
        self.direction = direction
        self.rec_iter = rec_iter
        # list of tuples of the form (object, distance)
        self.intersecting_objects: list[tuple[Surface, XYZ]] = []
        
    def compute_intersections(self, manual_surfaces=None, manual_materials=None):
        # fills up self.intersecting_objects
        # and then sorts it by distance
        materials = manual_materials if manual_materials is not None else self.materials
        surfaces_to_check = manual_surfaces if manual_surfaces is not None else self.surfaces
        for surface in surfaces_to_check:
            intersection_pnt = surface.intersect(self)
            if intersection_pnt is not None:
                self.intersecting_objects.append((surface, intersection_pnt))
        
        self.intersecting_objects.sort(key=lambda x: np.linalg.norm(self.origin - x[1]))
        if len(self.intersecting_objects) > self.max_recursions:
            self.intersecting_objects = self.intersecting_objects[:self.max_recursions]
        for i, (surface, _) in enumerate(self.intersecting_objects):
            material = materials[surface.material_index-1]
            if material.transparency == 0 and len(self.intersecting_objects) > i+1:
                self.intersecting_objects = self.intersecting_objects[:i+1]
                break
        
    def _compute_color(self) -> RGB:
        # returns the base color of the object
        # pre: assume `compute_intersections` has been called
        
        if len(self.intersecting_objects) == 0:
            return self.background_color
        
        def compute_color_rec(idx: int) -> np.ndarray:
            if idx >= len(self.intersecting_objects):
                return self.background_color
            
            surface = self.intersecting_objects[idx][0]
            if surface.material_index == -1:
                return self.background_color
            material = self.materials[surface.material_index - 1]
            
            surface_normal = surface.compute_normal(self.intersecting_objects[idx][1], self.direction)
            diffuse_color = self._compute_diffuse_light(surface_normal, surface, self.intersecting_objects[idx][1])
            specular_color = self._compute_specular_light(surface_normal, surface, self.intersecting_objects[idx][1])
            light_intensity = self._compute_light_intensity(surface, self.intersecting_objects[idx][1])

            res = (1 - material.transparency) * (diffuse_color + specular_color) * light_intensity
            if np.array(material.reflection_color).any(): #if reflection color is not black
                res += self._compute_reflection(surface_normal, surface, self.intersecting_objects[idx][1])
            if material.transparency > 0:
                res += material.transparency * np.array(compute_color_rec(idx + 1))
            return res
            
        return compute_color_rec(0)
    
    def _compute_diffuse_light(self, surface_normal: XYZ, surface: Surface, point_on_surface: XYZ) -> RGB:
        # returns the diffuse light of the object
        # pre: assume `compute_intersections` has been called
        N = surface_normal
        kd: RGB = self.materials[surface.material_index - 1].diffuse_color  
    
        res = np.zeros(3)
        for light in self.lights:
            I_L = np.array(light.color)
            L = light.position - point_on_surface
            L = L / np.linalg.norm(L)
            
            res += kd * I_L * max(0, N @ L) 

        return res
        
    def _compute_specular_light(self, surface_normal: XYZ, surface: Surface, point_on_surface: XYZ) -> RGB:
        # returns the specular light of the object
        # pre: assume `compute_intersections` has been called
        N = surface_normal
        ks: RGB = self.materials[surface.material_index - 1].specular_color
        n = self.materials[surface.material_index - 1].shininess
        
        res = np.zeros(3)
        for light in self.lights:
            I_L = np.array(light.color)
            L = light.position - point_on_surface
            L = L / np.linalg.norm(L)
            
            R = 2 * (N @ L) * N - L
            V = -self.direction
            
            res += (ks * I_L * max(0, R @ V) ** n)*light.specular_intensity
            
        return res
    
    def _compute_reflection(self, surface_normal: XYZ, surface: Surface, point_on_surface: XYZ) -> RGB:
        # returns the reflection of the object
        # pre: assume `compute_intersections` has been called
        if self.rec_iter >= self.max_recursions:
            return self.background_color
        
        N = surface_normal
        D = self.direction
        R = D - 2 * (N @ D) * N
        P = point_on_surface + 0.001 * N
        second_ray = Ray(P, R, self.rec_iter + 1)
        
        kr = np.array(self.materials[surface.material_index - 1].reflection_color)
        return kr * second_ray.trace_ray(self.scene_settings, self.surfaces, self.lights, self.materials)
    
    def _compute_light_intensity(self, surface: Surface, point_on_surface: XYZ) -> float:
        # returns the light intensity of the object
        # pre: assume `compute_intersections` has been called
        res = 0
        for light in self.lights:
            shadow_percentage = self._compute_shadow(light, surface, point_on_surface)
            res += (1 - light.shadow_intensity) * 1 + light.shadow_intensity * shadow_percentage
            #1 + intensity(percentage - 1)
        return res / len(self.lights)
    
    def _compute_shadow(self, light: Light, surface: Surface, point_on_surface: np.ndarray) -> float:
        """
        Computes the soft shadow factor for a given (surface, point_on_surface) lit by 'light'.
        We send N*N shadow rays from the light area to 'point_on_surface' and see how many
        are unblocked (i.e., hit 'surface' first).
        
        Returns a float in [0,1], where 0 means fully blocked, 1 means fully lit by that light.
        """
        N = self.shadow_ray_num
        if N == 0:
            # No shadow rays => fully lit
            return 1.0
        # The direction from the light center to the surface point
        main_dir = point_on_surface - light.position
        dist_to_surface = np.linalg.norm(main_dir)
        if dist_to_surface < 1e-9:
            # If the surface point is extremely close to the light, treat as fully lit
            return 1.0
        main_dir /= dist_to_surface  # normalize

        temp = np.array([1,0,0], dtype=float)
        if abs(main_dir @ temp) > 0.99:
            # if parallel or almost parallel, pick a different fallback
            temp = np.array([0,1,0], dtype=float)

        right = np.cross(main_dir, temp)
        right /= np.linalg.norm(right)
        up = np.cross(main_dir, right)
        up /= np.linalg.norm(up)

        radius = light.radius

        unblocked_count = 0
        plane_corner = light.position - radius/2 * right - radius/2 * up
        for j in range(N):
            for i in range(N):
                rand_x = np.random.rand() 
                rand_y = np.random.rand() 
                
                offset_x = ( (i + rand_x) / N)   # in [0, 1]
                offset_y = ( (j + rand_y) / N)   # in [0, 1]

                # world-space location of the temp light
                light_point = (plane_corner + offset_x * (radius) * right + offset_y * (radius) * up)

                # direction from the chosen temp light to the surface
                sub_dir = point_on_surface - light_point
                dist_sub = np.linalg.norm(sub_dir)
                if dist_sub < 1e-9:
                    # extremely close
                    unblocked_count += 1
                    continue
                sub_dir /= dist_sub

                # Build a shadow ray from 'light_point' -> 'point_on_surface'
                shadow_ray = Ray(light_point, sub_dir)
                shadow_ray.compute_intersections(self.surfaces, self.materials)

                # If the shadow ray has *any* intersection in front of 'light_point'
                # that is strictly closer than 'dist_sub' and is *not* the same surface at 'point_on_surface',
                # it is blocked. Otherwise unblocked.
                if not shadow_ray.intersecting_objects:
                    unblocked_count += 1
                else:
                    (hit_surface, hit_point) = shadow_ray.intersecting_objects[0]
                    hit_dist = np.linalg.norm(hit_point - light_point)

                    if (hit_surface == surface) and (abs(hit_dist - dist_sub) < 1e-4):
                        unblocked_count += 1
        shadow_factor = unblocked_count / float(N * N)
        return shadow_factor

        
    def trace_ray(self, scene_settings: SceneSettings, surfaces: list[Surface], lights: list[Light], materials: list[Material])->RGB:
        self.scene_settings = scene_settings #For passing to next rays in recursions
        self.max_recursions = int(scene_settings.max_recursions)
        self.background_color = tuple(scene_settings.background_color)
        self.shadow_ray_num = int(scene_settings.root_number_shadow_rays)

        self.surfaces = surfaces
        self.lights = lights
        self.materials = materials
        
        self.compute_intersections()
        color = self._compute_color()
        return color