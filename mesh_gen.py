import numpy as np
import pygmsh
import matplotlib.pyplot as plt
from physics import *

# Mesh Constants
NUM_ELEM_PUNCH = 24
NUM_ELEM_DIE = 100
PUNCH_REINFORCEMENT_MULTIPLE = 1
DIE_REINFORCEMENT_MULTIPLE = 3

#%%
def save_pairs(filename, data):
    np.savetxt(filename, data.T)

def load_pairs(filename):
    return np.loadtxt(filename).T

class Segment:
    def __init__(self, first_node_x, first_node_y, second_node_x, second_node_y):
        self.node1 = np.array([first_node_x, first_node_y]).reshape((2, 1))
        self.node2 = np.array([second_node_x, second_node_y]).reshape((2, 1))

class Circular_Arc:
    def __init__(self, center_x, center_y, radius, init_angle, final_angle):
        self.center = np.array([center_x, center_y]).reshape((2, 1))
        self.radius = radius
        self.angle_start = init_angle
        self.angle_end = final_angle
    def get_nodes(self):
        node1 = np.array([
            self.radius * np.cos(self.angle_start),
            self.radius * np.sin(self.angle_start)
        ]).reshape((2,1)) + self.center
        node2 = np.array([
            self.radius * np.cos(self.angle_end),
            self.radius * np.sin(self.angle_end)
        ]).reshape((2,1)) + self.center
        return np.concatenate((node1, node2), axis = 1)
#%%


#%%
def sheet_mesh_generation(
    # Sheet metal profile
    sheet_thickness, sheet_length, relative_groove_depth, nose_radius,
    # Options
    save_to_mem = False, load_from_mem = False
):
    """
    Return
    ------
    nodes : np.ndarray, shape(2, N)
        Coordinates of mesh nodes
    edges : np.ndarray, shape(2, M)
        Edges of mesh nodes, each column is a pair of indices (i, j) denoting connection between i-th and j-th elements in "nodes"
    boundary_edges : np.ndarray, shape(2, K)
        Edges along the boundary of the mesh, each column is a pair of indices (i, j) denoting connection between i-th and j-th elements in "nodes"
        Direction: Clock-wise
    surface_normal : np.ndarray, shape(2, K)
        Surface normal along the boundary of the mesh, each column is a pair of indices (i, j) denoting connection between i-th and j-th elements in "nodes". Has the same size as boundary_edges
    quads: np.ndarray, shape(L, 4)
        Each rows (i-th row) contains 4 node indices create the element
    """
    if load_from_mem:
        return load_pairs("sheet_nodes.txt"), load_pairs("sheet_connects.txt").astype(int), load_pairs("sheet_boundary_connects.txt").astype(int)

    with pygmsh.geo.Geometry() as geo:
        l = sheet_length
        t = sheet_thickness
        d = t * relative_groove_depth
        R_g = nose_radius
        V_groove_angle = np.pi / 2

        # Four corners
        # 0.22, 0.18
        point_bot_left = geo.add_point([-l/2, 0, 0], mesh_size=0.30)
        point_bot_right = geo.add_point([l/2, 0, 0], mesh_size=0.30)
        point_top_left = geo.add_point([-l/2, t, 0], mesh_size=0.30)
        point_top_right = geo.add_point([l/2, t, 0], mesh_size=0.30)

        # Starting Points for Groove Diagonals
        point_diag_left = geo.add_point([-(d + R_g*(np.sqrt(2)-1)), t, 0], mesh_size=0.25)
        point_diag_right = geo.add_point([d + R_g*(np.sqrt(2)-1), t, 0], mesh_size=0.25)

        # Points for Arc
        center = np.array([0, t - d + R_g])
        start = np.array([
            R_g*np.cos(-np.pi/2 + V_groove_angle/2),
            R_g*np.sin(-np.pi/2 + V_groove_angle/2)
        ]) + center
        end = np.array([
            R_g*np.cos(-np.pi/2 - V_groove_angle/2),
            R_g*np.sin(-np.pi/2 - V_groove_angle/2)
        ]) + center

        point_arc_center = geo.add_point(np.hstack((center, [0])), mesh_size=0.20)
        point_arc_start = geo.add_point(np.hstack((start, [0])), mesh_size=0.20)
        point_arc_end = geo.add_point(np.hstack((end, [0])), mesh_size=0.20)
        
        # Lines & Arc
        line_vert_left = geo.add_line(point_bot_left, point_top_left)
        line_top_left = geo.add_line(point_top_left, point_diag_left)
        line_diag_left = geo.add_line(point_diag_left, point_arc_end)
        arc = geo.add_circle_arc(point_arc_end, point_arc_center, point_arc_start)
        line_diag_right = geo.add_line(point_arc_start, point_diag_right)
        line_top_right = geo.add_line(point_diag_right, point_top_right)
        line_vert_right = geo.add_line(point_top_right, point_bot_right)
        line_bot = geo.add_line(point_bot_right, point_bot_left)

        # Loop
        loop = geo.add_curve_loop([
            line_vert_left,
            line_top_left,
            line_diag_left,
            arc,
            line_diag_right,
            line_top_right,
            line_vert_right,
            line_bot,
        ])

        geo.add_plane_surface(loop)

        # Mesh Generation
        mesh = geo.generate_mesh(dim=2, algorithm=11)

        # Initial Raw Data
        nodes = mesh.points[:, :2].T  # XY only
        elems = mesh.cells_dict['quad']
        raw_boundary_lines = mesh.cells_dict['line']

    # Mesh Sanitization (Remove Orphan Nodes)============
    
    # 1. Identify all unique nodes actually used by the quad elements
    unique_used_indices = np.unique(elems)
    
    # 2. Check if cleanup is needed
    if len(unique_used_indices) < nodes.shape[1]:
        # print(f"Sanitizing Mesh: Removed {nodes.shape[1] - len(unique_used_indices)} orphan nodes.")
        map_old_to_new = np.full(nodes.shape[1], -1, dtype=int) # initialize with -1
        map_old_to_new[unique_used_indices] = np.arange(len(unique_used_indices))
        nodes = nodes[:, unique_used_indices] # Filter the Nodes array
        elems = map_old_to_new[elems] # Remap the Quad Elements
        
        # 6. Remap and Filter Boundary Lines
        # GMSH sometimes returns lines connected to the orphan points (like the arc center).
        # We check if BOTH nodes of a line map to a valid new index.
        valid_boundary_mask = np.all(map_old_to_new[raw_boundary_lines] != -1, axis=1)
        boundary_edges = map_old_to_new[raw_boundary_lines[valid_boundary_mask]]
    else:
        boundary_edges = raw_boundary_lines

    # Transpose boundary to match your desired shape (2, K)
    boundary_edges = boundary_edges.T

    # Edges Generation ========================================
    first = elems[:, 0]
    second = elems[:, 1]
    third = elems[:, 2]
    fourth = elems[:, 3]
    edges = np.hstack((
        np.vstack((first, second)),
        np.vstack((second, third)),
        np.vstack((third, fourth)),
        np.vstack((fourth, first)),
    ))

    # Sort and remove duplicates to get unique edges
    sorted_edges = np.sort(edges, axis=0)
    edges = np.unique(sorted_edges, axis=1)

    if save_to_mem:
        save_pairs("sheet_nodes.txt", nodes)
        save_pairs("sheet_connects.txt", edges)
        save_pairs("sheet_boundary_connects.txt", boundary_edges)

    return nodes, edges, boundary_edges, elems
#%%


#%%
def punch_mesh_generation(
    # Main params
    punch_center, punch_angle, punch_radius, punch_height, num_elem,
    # Reinforments
    reinforce_multiple,
    # Options
    save_to_mem=False, load_from_mem=False
):
    if load_from_mem:
        return load_pairs("punch_nodes.txt"), load_pairs("punch_connects.txt").astype(int)

    # Geometry definition
    center_x = punch_center[0]
    center_y = punch_center[1]
    arc_angle = 180 - punch_angle
    arc = Circular_Arc(center_x, center_y, punch_radius, np.radians(-90 + arc_angle/2), np.radians(-90 - arc_angle/2))
    arc_ends = arc.get_nodes()
    line_angle1 = np.radians(90 - punch_angle/2)
    line_angle2 = np.radians(90 + punch_angle/2)
    height_limit = center_y + punch_height
    seg_right = Segment(
        arc_ends[0, 0],
        arc_ends[1, 0],
        arc_ends[0, 0] + np.cos(line_angle1) * (height_limit - arc_ends[1, 0] / np.sin(line_angle1)),
        height_limit
    )
    seg_left = Segment(
        arc_ends[0, 1],
        arc_ends[1, 1],
        arc_ends[0, 1] + np.cos(line_angle2) * (height_limit - arc_ends[1, 1] / np.sin(line_angle2)),
        height_limit
    )

    # Points generation
    num_nodes = [
        (num_elem + 1)//(2 + reinforce_multiple),
        np.ceil((num_elem + 1)/(2 + reinforce_multiple) * reinforce_multiple).astype(int),
        (num_elem + 1)//(2 + reinforce_multiple),
    ]
    if(np.sum(num_nodes) == num_elem): num_nodes[1] += 1
    nodes = np.array([[],[]])

    x_data = np.linspace(seg_left.node2[0], seg_left.node1[0], num_nodes[0], endpoint=False)
    y_data = np.linspace(seg_left.node2[1], seg_left.node1[1], num_nodes[0], endpoint=False)
    nodes = np.hstack((nodes, np.hstack((x_data, y_data)).T))

    angles = np.linspace(arc.angle_end, arc.angle_start, num_nodes[1], endpoint=False)
    for angle in angles:
        x = arc.center[0, 0] + arc.radius * np.cos(angle)
        y = arc.center[1, 0] + arc.radius * np.sin(angle)
        new_pair = np.array([[x], [y]])
        nodes = np.hstack((nodes, new_pair))

    x_data = np.linspace(seg_right.node1[0], seg_right.node2[0], num_nodes[2])
    y_data = np.linspace(seg_right.node1[1], seg_right.node2[1], num_nodes[2])
    nodes = np.hstack((nodes, np.hstack((x_data, y_data)).T))


    # Connections generation
    first = np.linspace(0, num_elem, num_elem, endpoint=False).astype(int)
    second = np.linspace(1, num_elem + 1, num_elem, endpoint=False).astype(int)
    edges = np.vstack((first, second))

    if save_to_mem:
        save_pairs("punch_nodes.txt", nodes)
        save_pairs("punch_connects.txt", edges)

    return nodes, edges

def get_punch_arc_ids(num_elem, reinforce_multiple):
    return [
        i for i in range(
            (num_elem + 1)//(2 + reinforce_multiple),
            (num_elem + 1)//(2 + reinforce_multiple) + np.ceil((num_elem + 1)/(2 + reinforce_multiple) * reinforce_multiple).astype(int) + 1
        )
    ]
#%%

#%%
def die_mesh_generation(die_angle, die_groove_length, curve_radius, die_length, num_elem, reinforce_multiple):
    # Geometry definition
    seg_left = Segment(-die_length/2, 0, -die_groove_length/2, 0)
    seg_right = Segment(die_groove_length/2, 0, die_length/2, 0)

    arc_angle = np.radians(90 - die_angle / 2)
    arc_center_left = (-die_groove_length/2, -curve_radius)
    arc_center_right = (die_groove_length/2, -curve_radius)
    arc_left = Circular_Arc(arc_center_left[0], arc_center_left[1], curve_radius, np.radians(90), np.radians(90) - arc_angle)
    arc_right = Circular_Arc(arc_center_right[0], arc_center_right[1], curve_radius, np.radians(90) + arc_angle, np.radians(90))

    arc_left_points = arc_left.get_nodes()
    arc_right_points = arc_right.get_nodes()
    seg_diag_left = Segment(arc_left_points[0, 1], arc_left_points[1, 1], 0, - (die_groove_length / 2) / np.tan(np.radians(die_angle / 2)))
    seg_diag_right = Segment(0, - (die_groove_length / 2) / np.tan(np.radians(die_angle / 2)), arc_right_points[0, 0], arc_right_points[1, 0])

    # Node generation
    nodes = np.array([[], []])
    num_nodes = [
        (num_elem + 1)//(2 + reinforce_multiple),
        np.ceil((num_elem + 1)/(2 + reinforce_multiple) * reinforce_multiple).astype(int),
        (num_elem + 1)//(2 + reinforce_multiple),
    ]
    if(np.sum(num_nodes) == num_elem): num_nodes[1] += 1

    x_data = np.linspace(seg_left.node1[0], seg_left.node2[0], num_nodes[0], endpoint=False)
    y_data = np.linspace(seg_left.node1[1], seg_left.node2[1], num_nodes[0], endpoint=False)
    nodes = np.hstack((nodes, np.hstack((x_data, y_data)).T))

    x_data = np.linspace(arc_left_points[0, 0], arc_right_points[0, 1], num_nodes[1], endpoint=False).reshape((num_nodes[1], 1))
    y_data = np.array([])
    for x in x_data:
        if x < arc_left_points[0, 1]: # Check lies in left arc
            angle = np.acos((x - arc_left.center[0]) / arc_left.radius)
            y = arc_left.radius * np.sin(angle) + arc_left.center[1]
        elif x < 0: # Check lies in left diagonal
            p1, p2 = seg_diag_left.node1, seg_diag_left.node2
            t = (x - p1[0]) / (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
        elif x < seg_diag_right.node2[0]: # Check lies in right diagonal
            p1, p2 = seg_diag_right.node1, seg_diag_right.node2
            t = (x - p1[0]) / (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
        else: # Lies in left arc
            angle = np.acos((x - arc_right.center[0]) / arc_right.radius)
            y = arc_right.radius * np.sin(angle) + arc_right.center[1]
        y_data = np.append(y_data, y, axis=0)

    y_data = y_data.reshape((num_nodes[1], 1))
    nodes = np.hstack((nodes, np.hstack((x_data, y_data)).T))

    x_data = np.linspace(seg_right.node1[0], seg_right.node2[0], num_nodes[2])
    y_data = np.linspace(seg_right.node1[1], seg_right.node2[1], num_nodes[2])
    nodes = np.hstack((nodes, np.hstack((x_data, y_data)).T))

    # Edge generation
    edges = np.array([[], []])
    for i in range(num_elem):
        edges = np.hstack((edges, [[i], [i+1]]))

    return nodes, edges.astype(int)
#%%


#%%
if __name__ == "__main__":
    sheet_nodes, sheet_edges, sheet_boundaries = sheet_mesh_generation(THICKNESS, LENGTH, RELATIVE_GROOVE_DEPTH, NOSE_RADIUS)
    punch_nodes, punch_edges = punch_mesh_generation(PUNCH_POSITION, PUNCH_ANGLE, PUNCH_RADIUS, PUNCH_HEIGHT, NUM_ELEM_PUNCH, PUNCH_REINFORCEMENT_MULTIPLE)
    die_nodes, die_edges = die_mesh_generation(DIE_ANGLE, DIE_GROOVE_LENGTH, DIE_CURVE_RADIUS, LENGTH, NUM_ELEM_DIE, DIE_REINFORCEMENT_MULTIPLE)

    for idx in range(sheet_edges.shape[1]):
        first = sheet_edges[0, idx]
        second = sheet_edges[1, idx]
        plt.plot([sheet_nodes[0, first], sheet_nodes[0, second]], [sheet_nodes[1, first], sheet_nodes[1, second]], color="orange")

    for idx in range(punch_edges.shape[1]):
        first = punch_edges[0, idx]
        second = punch_edges[1, idx]
        plt.plot([punch_nodes[0, first], punch_nodes[0, second]], [punch_nodes[1, first], punch_nodes[1, second]], color="blue")

    for idx in range(die_edges.shape[1]):
        first = die_edges[0, idx]
        second = die_edges[1, idx]
        plt.plot([die_nodes[0, first], die_nodes[0, second]], [die_nodes[1, first], die_nodes[1, second]], color="green")

    plt.axis('equal')
    plt.show()
#%%

