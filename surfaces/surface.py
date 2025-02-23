from abc import ABC, abstractmethod
import numpy as np

class Surface(ABC):
    
    def __init__(self, material_index: int):
        self.material_index = material_index
        
    @abstractmethod
    def intersect(self, ray):
        pass
    
    @abstractmethod
    def compute_normal(self, point_on_surface, ray_direction) -> np.ndarray:
        pass
