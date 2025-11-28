import numpy as np
import matplotlib.pyplot as plt
from maths import *

STANDARD_GRID_SIZE = (1, 1) # mm
DIVISION = 8
GRID_SIZE = (STANDARD_GRID_SIZE[0] / DIVISION, STANDARD_GRID_SIZE[1] / DIVISION)
COORDINATE_MIN = (-np.ceil(LENGTH/2 + 0.01), -np.ceil((DIE_GROOVE_LENGTH/2)/np.tan(np.radians(DIE_ANGLE/2))))
COORDINATE_MAX = (np.ceil(LENGTH/2 + 0.01), np.ceil(PUNCH_HEIGHT+0.1)) # mm


class SpatialHash2D:
    def __init__(self, coord_min, coord_max, grid_size):
        self.coord_min = np.array(coord_min)
        self.coord_max = np.array(coord_max)
        self.grid_size = np.array(grid_size)
        self.nx = int(np.ceil((coord_max[0] - coord_min[0]) / grid_size[0]))
        self.ny = int(np.ceil((coord_max[1] - coord_min[1]) / grid_size[1]))
        self.cells = {}  # dictionary mapping (i, j) → [node_indices]

    def hash(self, point):
        """Convert physical coordinates to cell indices."""
        i = int(np.floor((point[0] - self.coord_min[0]) / self.grid_size[0]))
        j = int(np.floor((point[1] - self.coord_min[1]) / self.grid_size[1]))
        return (i, j)

    # def build(self, *object_nodes):
    #     """
    #     Build the hash grid from multiple objects.
    #     Each argument = 2xN numpy array (node coordinates of an object).
    #     """
    #     self.cells.clear()
    #     for obj_id, nodes in enumerate(object_nodes):
    #         for idx in range(nodes.shape[1]):
    #             i, j = self.hash(nodes[0, idx], nodes[1, idx])
    #             self.cells.setdefault((i, j), []).append((obj_id, idx))

    def build(self, *bodies):
        """
        Build grid for multiple bodies.
        Each body can be either:
          (coords, indices)  -> only store selected nodes
          (coords,)          -> store all nodes
        """
        self.cells.clear()

        for body_id, body in enumerate(bodies):
            # Unpack body tuple
            if len(body) == 2:
                coords, indices = body
                node_indices = indices
            else:
                raise ValueError("Each body must be (coords,) or (coords, indices).")

            # Insert selected points into hash cells
            for node_id in node_indices:
                p = coords[:, node_id]
                key = self.hash(p)
                self.cells.setdefault(key, []).append((body_id, node_id))

    def query_neighbors(self, point, exclude_obj_id=None):
        """
        Return
        -------
        neighbors: a dictionary of nearby nodes
            neighbors[i] = j
            denoting the j-th node in the i-th body
        """
        i, j = self.hash(point)
        neighbors = {}
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                cell = (i + di, j + dj)
                if cell in self.cells:
                    for (obj_id, idx) in self.cells[cell]:
                        if exclude_obj_id is None or obj_id not in exclude_obj_id:
                            neighbors.setdefault(obj_id, []).append(idx)
        return neighbors
    
    def draw(self):
        top_points = np.vstack((
            np.linspace(self.coord_min[0], self.coord_max[0], self.nx + 1),
            np.ones(self.nx + 1) * self.coord_max[1]
        ))
        bottom_points = np.vstack((
            np.linspace(self.coord_min[0], self.coord_max[0], self.nx + 1),
            np.ones(self.nx + 1) * self.coord_min[1]
        ))
        left_points = np.vstack((
            np.ones(self.ny + 1) * self.coord_min[0],
            np.linspace(self.coord_min[1], self.coord_max[1], self.ny + 1)
        ))
        right_points = np.vstack((
            np.ones(self.ny + 1) * self.coord_max[0],
            np.linspace(self.coord_min[1], self.coord_max[1], self.ny + 1)
        ))

        for i in range(self.nx + 1):
            plt.plot([top_points[0, i], bottom_points[0, i]], [top_points[1, i], bottom_points[1, i]], color="#D3D3D3")

        for i in range(self.ny + 1):
            plt.plot([left_points[0, i], right_points[0, i]], [left_points[1, i], right_points[1, i]], color="#D3D3D3")
