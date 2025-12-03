import numpy as np
from pygmsh.common import point

# Material Dimensions
THICKNESS, LENGTH, WIDTH = 1.2, 16, 32 #mm
RELATIVE_GROOVE_DEPTH, NOSE_RADIUS = 0.5, 0.8 # %, mm

# Punch, Die Dimensions
PUNCH_POSITION = (0, 1.2) #mm
PUNCH_ANGLE, PUNCH_RADIUS, PUNCH_HEIGHT = 86, 0.6, 2 #deg, mm, mm
DIE_ANGLE = 86
DIE_GROOVE_LENGTH = 12 #mm
DIE_CURVE_RADIUS = 2.5 #mm

PENETRATION_ANGLE_THRESHOLD = 30

def find_gaussian_points_on_boundary(nodes, boundary_edges, num_gauss_points):
    """
    Generate Gaussian points and their corresponding weights on boundary segments.

    Parameters
    ----------
    nodes : np.ndarray, shape (2, N)
        Coordinates of original mesh nodes.
    boundary_edges : np.ndarray, shape (2, M)
        Each column gives the node indices (0-based) for a boundary edge.
    num_gauss_points : int
        Number of Gauss points per edge.

    Returns
    -------
    gauss_nodes : np.ndarray, shape (2, M * num_gauss_points)
        Coordinates of all Gaussian points (global coordinates).
    mapping : dict
        mapping[i] = (segment_index, weight)
        where i is the index (0-based) of the Gaussian point in gauss_nodes.
    """
    # 1. Gauss–Legendre quadrature points and weights on [-1, 1]
    xi, w = np.polynomial.legendre.leggauss(num_gauss_points)
    N1 = (1 - xi) / 2
    N2 = (1 + xi) / 2

    gauss_nodes_list = []
    mapping = {}
    idx = 0

    # 2. Loop through each boundary segment
    for seg_idx in range(boundary_edges.shape[1]):
        n1, n2 = boundary_edges[:, seg_idx]
        x1, x2 = nodes[:, n1], nodes[:, n2]

        # Compute Gauss points in global coordinates
        xg = np.outer(x1, N1) + np.outer(x2, N2)  # shape (2, num_gauss_points)

        for k in range(num_gauss_points):
            gauss_nodes_list.append(xg[:, k])
            mapping[idx] = (seg_idx, xi[k], w[k])
            idx += 1

    # 3. Stack all Gaussian nodes into one array
    gauss_nodes = np.stack(gauss_nodes_list, axis=1) if gauss_nodes_list else np.zeros((2, 0))

    return gauss_nodes, mapping

def find_closest_point(ref_point, points, indices):
    """
    Find the index (from `indices`) of the point in `points` closest to `ref_point`.

    Parameters
    ----------
    ref_point : array-like, shape (2,)
        The reference point [x, y].
    points : np.ndarray, shape (2, N)
        Array of 2D points (each column is a point).
    indices : list or np.ndarray
        List of indices (0-based) into `points` to consider.

    Returns
    -------
    int
        The index (from `indices`) of the closest point to `ref_point`.
    """
    subset = points[:, indices]
    dists2 = np.sum((subset.T - ref_point)**2, axis=1)
    closest_local_idx = np.argmin(dists2)
    return indices[closest_local_idx]

def find_projection(ref_point, points, segments, indices):
    """
    Compute the orthogonal projection of a reference point onto the closest segment.

    Parameters
    ----------
    ref_point : np.ndarray, shape (2,)
        The reference point in 2D space.
    points : np.ndarray, shape (2, N)
        Coordinates of all mesh nodes (each column is a node [x, y]).
    segments : np.ndarray, shape (2, M)
        Each column defines a line segment by storing two node indices.
    indices : list or np.ndarray
        Indices (0-based) of the segments to consider for the search.

    Returns
    -------
    closest_seg_idx : int
        Index (from `segments`) of the closest segment to the reference point.
    t_value : float
        Local coordinate along that segment (0 at the first node, 1 at the second).
    proj_point : np.ndarray, shape (2,)
        Coordinates of the orthogonal projection of `ref_point` onto that segment.

    Notes
    -----
    - If the perpendicular projection lies outside the segment, the projection
      is clamped to the nearest endpoint.
    - Useful for surface contact detection, distance computation, and
      master-slave projection in 2D FEM contact algorithms.
    """
    min_dist2 = np.inf
    closest_seg_idx = None
    t_value = None
    proj_coord = None

    for seg_idx in indices:
        if len(seg_idx) == 0: continue
        n1, n2 = segments[0, seg_idx], segments[1, seg_idx]
        p1, p2 = points[:, n1].reshape(2), points[:, n2].reshape(2)
        v = p2 - p1
        w = ref_point - p1
        t = np.dot(w.T, v) / np.sum(v**2)

        if t >= 1:
            t = 1
            proj = p2
        elif t <= 0:
            t = 0
            proj = p1
        else:
            proj = p1 + t * v

        dist2 = np.sum((proj - ref_point)**2)

        if dist2 < min_dist2:
            min_dist2 = dist2
            closest_seg_idx = seg_idx
            t_value = t
            proj_coord = proj

    return closest_seg_idx, t_value, proj_coord

def find_surface_normal(nodes, boundary_edges):
    p_i = nodes[:, boundary_edges[0, :]]
    p_j = nodes[:, boundary_edges[1, :]]
    vec_ij = p_j - p_i
    normals = np.vstack((-vec_ij[1, :], vec_ij[0, :])) # Rotate 90° counterclockwise
    lengths = np.linalg.norm(normals, axis=0, keepdims=True)
    return normals / lengths

def find_area_element_Q4(nodes):
    """
    Compute the area of a 4-node quadrilateral element using the shoelace formula.
    
    Parameters
    ----------
    nodes : np.ndarray, shape (4, 2)
        Each row is a 2D coordinate [x, y] of a vertex, ordered around the element.
    
    Returns
    -------
    float
        Area of the quadrilateral.
    """
    x = nodes[:, 0]
    y = nodes[:, 1]
    # Use np.roll to wrap around
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area

def integrate_gauss_2D(func, nodes, nodal_disp, num_gp_per_side):
    """
    Integrate the function "func" over the range inside finitesimal elemnt bounded by 4 nodes in "nodes"

    Parameters
    ----------
    func: python function
        any function that
            - takes 2 values: xi and eta
    nodes : np.ndarray, shape (4, 2)
        nodal coordinates of Q4 element
        IMPORTANT: the order of nodes MUST satisfy counter-clockwise direction
    nodal_disp: np.ndarray, shape(8, 1)
        current nodal displacement of the element
    num_gp_per_side : int
        Number of quadrature points per side.

    Returns
    ----------
    result: the resultant integration
        np.ndarray with shape depends on the input "func"

    """
    # 1D Gauss-Legendre points and weights
    xi_1D, w_1D = np.polynomial.legendre.leggauss(num_gp_per_side)

    # Create tensor product for 2D
    xis = np.array([xi for eta in xi_1D for xi in xi_1D])
    etas = np.array([eta for eta in xi_1D for xi in xi_1D])
    weights = np.array([wx * wy for wy in w_1D for wx in w_1D])

    # Initialize result (based on function output)
    sample = func(xis[0], etas[0])
    result = np.zeros_like(sample)

    for i in range(num_gp_per_side**2):
        xi, eta = xis[i], etas[i]
        w = weights[i]
        der = 0.25 * np.array([
            [-(1-eta), (1-eta), (1+eta), -(1+eta)],
            [-(1-xi), -(1+xi), (1+xi), (1-xi)]
        ])
        Jacobian = der @ nodes
        result += w * func(xi, eta, J=Jacobian, der=der, u_e=nodal_disp) * abs(np.linalg.det(Jacobian))

    return result


