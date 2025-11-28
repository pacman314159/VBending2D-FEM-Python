import numpy as np
import matplotlib.pyplot as plt
from mesh_gen import *
from grid_hash import *
from maths import *
from physics import *

# Animation Constants
NON_CONTACT_TIME_STEP = 0.001 #s
FEA_TIME_STEP = 1e-4 #s

# For Plane Strain Problem
D_MATRIX = np.array([
    [1 - POISSON_RATIO, -POISSON_RATIO, 0],
    [-POISSON_RATIO, 1-POISSON_RATIO, 0],
    [0, 0, 0.5 * (1 - 2 * POISSON_RATIO)]
]) * YOUNG_MODULUS / (1 + POISSON_RATIO) / (1 - 2 * POISSON_RATIO)


if __name__ == "__main__":
    # GEOMETRY PREPARATIONS --------------------------------------------------------------------
    sheet_nodes, sheet_edges, sheet_boundary, sheet_elems = sheet_mesh_generation(THICKNESS, LENGTH, RELATIVE_GROOVE_DEPTH, NOSE_RADIUS)
    sheet_gp, sheet_gp_dict = find_gaussian_points_on_boundary(sheet_nodes, sheet_boundary, NUM_GP_CONTACT)
    punch_nodes, punch_edges = punch_mesh_generation(
        PUNCH_POSITION, PUNCH_ANGLE, PUNCH_RADIUS, PUNCH_HEIGHT,
        NUM_ELEM_PUNCH, PUNCH_REINFORCEMENT_MULTIPLE,
    )
    punch_arc_ids = get_punch_arc_ids(NUM_ELEM_PUNCH, PUNCH_REINFORCEMENT_MULTIPLE)
    die_nodes, die_edges = die_mesh_generation(
        DIE_ANGLE, DIE_GROOVE_LENGTH, DIE_CURVE_RADIUS, LENGTH,
        NUM_ELEM_DIE, DIE_REINFORCEMENT_MULTIPLE
    )
    nodes_of_body = {
        0: punch_nodes,
        1: sheet_gp,
        2: die_nodes
    }
    edges_of_body = {
        0: punch_edges,
        2: die_edges
    }
    grid = SpatialHash2D(COORDINATE_MIN, COORDINATE_MAX, GRID_SIZE)

    # 1. Identify all nodes actually used by elements
    used_node_indices = set()
    for elem in sheet_elems:
        used_node_indices.update(elem)
    
    # 2. Check if we have orphans
    total_nodes = sheet_nodes.shape[1]
    if len(used_node_indices) < total_nodes:
        print(f"Found {total_nodes - len(used_node_indices)} orphan nodes!")

    # CLF_LIMIT_SEC = get_CFL_limit(sheet_nodes, YOUNG_MODULUS * 1e9, POISSON_RATIO, DENSITY * 1e9)
    
    t = 0
    num_loops = 1
    time_step = NON_CONTACT_TIME_STEP
    Displacement_global = np.zeros(sheet_nodes.shape[1] * 2)

    for loop in range(num_loops):
        punch_nodes += PUNCH_VELOCITY * time_step 
        t = t + time_step

#%% Calculate Contact & Friction -----------------------------------------------------
        F_contact = np.zeros(sheet_nodes.shape[1] * 2) # positive: Point away from body (reaction force)
        K_contact = np.zeros((sheet_nodes.shape[1] * 2, sheet_nodes.shape[1] * 2))
        sheet_surface_normals = find_surface_normal(sheet_nodes, sheet_boundary)
        grid.build(
            (punch_nodes, np.arange(NUM_ELEM_PUNCH+1).astype(int)),
            (sheet_gp, np.arange(sheet_gp.shape[1]).astype(int)),
            (die_nodes, np.arange(NUM_ELEM_DIE+1).astype(int))
        )

        for gp_id, (seg_id, xi, gauss_weight) in sheet_gp_dict.items():
            gp = sheet_gp[:, gp_id]
            surface_normal = sheet_surface_normals[:, seg_id]

            # Find closest node on punch relative to gauss point
            neighbors = grid.query_neighbors(gp, exclude_obj_id=[1]) # Reduce the finding space
            if len(neighbors) == 0 : continue # skip through "lonely" nodes
            for body_id in neighbors.keys():
                master_nodes = nodes_of_body[body_id]
                master_edges = edges_of_body[body_id]

                closest_node_id = find_closest_point(gp, master_nodes, neighbors[body_id])

                # Find closest segment relative to the gauss point
                first_seg_id = np.where(master_edges[0] == closest_node_id)[0]
                second_seg_id = np.where(master_edges[1] == closest_node_id)[0]
                closest_seg_id, _, proj = find_projection(gp, master_nodes, master_edges, [first_seg_id, second_seg_id])

                plt.plot([gp[0], proj[0]], [gp[1], proj[1]], color='blue')
                
                # Contact pressure & traction
                gap_normal = np.dot((proj - gp), surface_normal)
                if gap_normal >= 0: continue # Proceed if penetration detected

                time_step = FEA_TIME_STEP
                contact_pressure = - PENALTY_STIFFNESS_NORMAL * gap_normal
                traction = contact_pressure * surface_normal

                # Assemble nodal forces
                N1, N2 = (1 - xi) / 2.0, (1 + xi) / 2.0
                id1, id2 = sheet_boundary[:, seg_id]
                p1, p2 = sheet_nodes[:, id1], sheet_nodes[:, id2]
                segment_length = np.linalg.norm(p2 - p1)
                f1c = N1 * gauss_weight * traction * segment_length / 2.0
                f2c = N2 * gauss_weight * traction * segment_length / 2.0
                F_contact[id1*2] += f1c[0]
                F_contact[id2*2] += f2c[0]
                F_contact[id1*2+1] += f1c[1]
                F_contact[id2*2+1] += f2c[1]

#%%

#%% Calculate Body Forces -----------------------------------------------------
        F_body = np.zeros(sheet_nodes.shape[1] * 2)
        for first, fourth, third, second in sheet_elems: # Ensure direction is CCW
            elem_nodes = sheet_nodes[:, [first, second, third, fourth]].T # (4, 2) matrix
            elem_dofs = np.array([first*2, first*2+1, second*2, second*2+1, third*2, third*2+1, fourth*2, fourth*2+1]) # Global index map
            f_body_elem = integrate_gauss_2D(
                elem_body_force_func,
                elem_nodes,
                None,
                NUM_GP_2D_ELEM_PER_SIDE
            ) * WIDTH
            F_body[elem_dofs] += f_body_elem
#%%

#%% Calculate Internal Forces -----------------------------------------------------
        F_internal = np.zeros(sheet_nodes.shape[1] * 2)
        for first, fourth, third, second in sheet_elems: # Ensure direction is CCW
            elem_nodes = sheet_nodes[:, [first, second, third, fourth]].T # (4, 2) matrix
            elem_dofs = np.array([first*2, first*2+1, second*2, second*2+1, third*2, third*2+1, fourth*2, fourth*2+1]) # Global index map
            u_e = Displacement_global[elem_dofs]
            f_internal = integrate_gauss_2D(
                elem_internal_force_func,
                elem_nodes,
                u_e,
                NUM_GP_2D_ELEM_PER_SIDE
            ) * WIDTH
            F_internal[elem_dofs] += f_internal
#%%

#%% Assemble material stiffness matrix
        K_material = np.zeros((sheet_nodes.shape[1]*2, sheet_nodes.shape[1]*2))
        for first, fourth, third, second in sheet_elems: # Ensure direction is CCW
            elem_nodes = sheet_nodes[:, [first, second, third, fourth]].T # (4, 2) matrix
            elem_dofs = np.array([first*2, first*2+1, second*2, second*2+1, third*2, third*2+1, fourth*2, fourth*2+1]) # Global index map
            k_e = integrate_gauss_2D(
                elem_stiffness_matrix_func,
                elem_nodes,
                None,
                NUM_GP_2D_ELEM_PER_SIDE
            ) * WIDTH
            for a in range(8):
                for b in range(8):
                    A = elem_dofs[a]
                    B = elem_dofs[b]
                    K_material[A, B] += k_e[a, b]

#%%

#%% Assemble geometric stiffness matrix
        K_geometric = np.zeros((sheet_nodes.shape[1]*2, sheet_nodes.shape[1]*2))
        for first, fourth, third, second in sheet_elems: # Ensure direction is CCW
            elem_nodes = sheet_nodes[:, [first, second, third, fourth]].T # (4, 2) matrix
            elem_dofs = np.array([first*2, first*2+1, second*2, second*2+1, third*2, third*2+1, fourth*2, fourth*2+1]) # Global index map
            u_e = Displacement_global[elem_dofs]
            k_e = integrate_gauss_2D(
                elem_geometric_matrix_func,
                elem_nodes,
                u_e,
                NUM_GP_2D_ELEM_PER_SIDE
            ) * WIDTH
            for a in range(8):
                for b in range(8):
                    A = elem_dofs[a]
                    B = elem_dofs[b]
                    K_geometric[A, B] += k_e[a, b]
#%%


        F_external = F_body - F_contact
        K_tangent = K_material + K_geometric
        Residual = F_external - F_internal

        left_node = np.argmin(sheet_nodes[0])           # index of leftmost node
        right_node = np.argmax(sheet_nodes[0])          # index of rightmost node

        print(left_node)
        fixed_dofs = [
            left_node*2, left_node*2+1,      # fix left node ux,uy
            right_node*2, right_node*2+1
        ]
        apply_dirichlet(K_tangent, Residual, fixed_dofs)

        delta_u = np.linalg.solve(K_tangent, Residual)
        print(delta_u)


    for pair in sheet_edges.T:
        first, second = pair[0], pair[1]
        x_data = [sheet_nodes[0][first], sheet_nodes[0][second]]
        y_data = [sheet_nodes[1][first], sheet_nodes[1][second]]
        plt.plot(x_data, y_data, color="orange")

    for pair in punch_edges.T:
        first, second = pair[0], pair[1]
        x_data = [punch_nodes[0][first], punch_nodes[0][second]]
        y_data = [punch_nodes[1][first], punch_nodes[1][second]]
        plt.plot(x_data, y_data, color="brown")

    for pair in die_edges.T:
        first, second = pair[0], pair[1]
        x_data = [die_nodes[0][first], die_nodes[0][second]]
        y_data = [die_nodes[1][first], die_nodes[1][second]]
        plt.plot(x_data, y_data, color="brown")

    plt.scatter(sheet_nodes[0, [left_node, right_node]], sheet_nodes[1, [left_node, right_node]], c="red")

    # grid.draw()

    plt.xlim(COORDINATE_MIN[0], COORDINATE_MAX[0])
    plt.ylim(COORDINATE_MIN[1], COORDINATE_MAX[1])
    plt.axis('equal')
    plt.show()
