import os
import numpy as np
import matplotlib.pyplot as plt
from mesh_gen import *
from grid_hash import *
from maths import *
from physics import *
from animator import save_animation

NUM_LOOPS = 500
TIME_STEP = 1e-3 #seconds
MAX_NEWTON_ITERATIONS = 15
NEWTON_RAPHSON_TOLERANCE = 1e0

#%%
def visualize(sheet_nodes, sheet_edges, punch_nodes, punch_edges, die_nodes, die_edges):
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

    # grid.draw()

    plt.xlim(COORDINATE_MIN[0], COORDINATE_MAX[0])
    plt.ylim(COORDINATE_MIN[1], COORDINATE_MAX[1])
    plt.axis('equal')
    plt.show()
#%%

def assemble_and_solve(
    sheet_nodes, sheet_elems, sheet_boundary, 
    punch_nodes, punch_edges, die_nodes, die_edges,
    Disp_global, grid, fixed_dofs
):

#%% 1. UPDATE GEOMETRY
    current_sheet_nodes = sheet_nodes + Disp_global.reshape((2, -1), order='F')
    sheet_gp, sheet_gp_dict = find_gaussian_points_on_boundary(current_sheet_nodes, sheet_boundary, NUM_GP_CONTACT)
    sheet_surface_normals = find_surface_normal(current_sheet_nodes, sheet_boundary)
    nodes_of_body = {0: punch_nodes, 1: sheet_gp, 2: die_nodes}
    edges_of_body = {0: punch_edges, 2: die_edges}

    grid.build(
        (punch_nodes, np.arange(NUM_ELEM_PUNCH+1).astype(int)),
        (sheet_gp, np.arange(sheet_gp.shape[1]).astype(int)),
        (die_nodes, np.arange(NUM_ELEM_DIE+1).astype(int))
    )
#%%

#%% 2. MATRIX INITIALIZATION
    F_body      = np.zeros(Total_dofs)
    F_internal  = np.zeros(Total_dofs)
    F_contact   = np.zeros(Total_dofs) # positive: Point away from body (reaction force)
    K_material  = np.zeros((Total_dofs, Total_dofs))
    K_geometric = np.zeros((Total_dofs, Total_dofs))
    K_contact   = np.zeros((Total_dofs, Total_dofs))
#%%

#%% 3. CONTACT DETECTION, FORCES & STIFFNESS MATRIX
    for gp_id, (seg_id, xi, gauss_weight) in sheet_gp_dict.items():
        gp = sheet_gp[:, gp_id]
        surface_normal = sheet_surface_normals[:, seg_id]

        # Find closest node on punch relative to gauss point
        neighbors = grid.query_neighbors(gp, exclude_obj_id=[1]) # Reduce the finding space
        if len(neighbors) == 0: continue # skip through "lonely" nodes
        for body_id in neighbors.keys():
            master_nodes = nodes_of_body[body_id]
            master_edges = edges_of_body[body_id]

            closest_node_id = find_closest_point(gp, master_nodes, neighbors[body_id])

            # Find closest segment relative to the gauss point
            first_seg_id = np.where(master_edges[0] == closest_node_id)[0]
            second_seg_id = np.where(master_edges[1] == closest_node_id)[0]
            closest_seg_id, _, proj = find_projection(gp, master_nodes, master_edges, [first_seg_id, second_seg_id])
            
            # Penetration detection
            gap_normal = np.dot((proj - gp), surface_normal)
            if gap_normal >= 0: continue
            angle_between = np.degrees(np.arctan2((proj - gp)[1], (proj - gp)[0]) - np.arctan2(surface_normal[1], surface_normal[0])) % 360
            if not (180 - PENETRATION_ANGLE_THRESHOLD/2 <= angle_between <= 180 + PENETRATION_ANGLE_THRESHOLD/2):
                continue
            
            contact_pressure = PENALTY_STIFFNESS_NORMAL * np.abs(gap_normal)
            traction = contact_pressure * surface_normal

            # Assemble nodal forces
            N_vals = [(1 - xi) / 2.0, (1 + xi) / 2.0] # Shape functions
            node_ids = sheet_boundary[:, seg_id]
            elem_dofs = np.array([node_ids[0]*2, node_ids[0]*2+1, node_ids[1]*2, node_ids[1]*2+1])
            p1 = current_sheet_nodes[:, node_ids[0]]
            p2 = current_sheet_nodes[:, node_ids[1]]
            seg_len = np.linalg.norm(p2 - p1)
            f = np.outer(N_vals, traction).flatten()
            F_contact[elem_dofs] += f * gauss_weight * seg_len * 0.5

            # Assemble contact stiffness matrix
            k_coeff = PENALTY_STIFFNESS_NORMAL * gauss_weight * seg_len * 0.5 
            n_mat = np.outer(surface_normal, surface_normal) 
            for a in range(2): 
                for b in range(2):
                    w_stiff = k_coeff * N_vals[a] * N_vals[b]
                    k_block = w_stiff * n_mat
                    r_x, r_y = elem_dofs[2*a], elem_dofs[2*a+1] # Global Row indices (u, v) for node a
                    c_x, c_y = elem_dofs[2*b], elem_dofs[2*b+1] # Global Column indices (u, v) for node b
                    K_contact[r_x, c_x] += k_block[0, 0]
                    K_contact[r_x, c_y] += k_block[0, 1]
                    K_contact[r_y, c_x] += k_block[1, 0]
                    K_contact[r_y, c_y] += k_block[1, 1]
#%%

#%% 4. BODY, INTERNAL FORCES & MATERIAL, GEOMETRIC STIFFNESS MATRICES
    for first, fourth, third, second in sheet_elems: # Ensure direction is CCW
        elem_nodes = sheet_nodes[:, [first, second, third, fourth]].T # (4, 2) matrix
        elem_dofs = np.array([first*2, first*2+1, second*2, second*2+1, third*2, third*2+1, fourth*2, fourth*2+1]) # Global index map
        u_e = Disp_global[elem_dofs]

        # Integration
        f_body = integrate_gauss_2D(body_force_func, elem_nodes, None, NUM_GP_2D_ELEM_PER_SIDE) * WIDTH
        f_int = integrate_gauss_2D(internal_force_func, elem_nodes, u_e, NUM_GP_2D_ELEM_PER_SIDE) * WIDTH
        k_mat = integrate_gauss_2D(material_stiffness_matrix_func, elem_nodes, u_e, NUM_GP_2D_ELEM_PER_SIDE) * WIDTH
        k_geom = integrate_gauss_2D(geometric_matrix_func, elem_nodes, u_e, NUM_GP_2D_ELEM_PER_SIDE) * WIDTH

        # Assembly
        F_body[elem_dofs] += f_body
        F_internal[elem_dofs] += f_int
        K_material[np.ix_(elem_dofs, elem_dofs)] += k_mat
        K_geometric[np.ix_(elem_dofs, elem_dofs)] += k_geom
#%%

#%% 5. SOLVE
    F_external = F_body - F_contact
    K_tangent = K_material + K_geometric + K_contact
    Residual = F_external - F_internal

    apply_dirichlet(K_tangent, Residual, fixed_dofs)

    Delta_u = np.linalg.solve(K_tangent, Residual)
    res_norm = np.linalg.norm(Residual)

    return Delta_u, res_norm
#%%


#%%
if __name__ == "__main__":

    # 1. SETUP HISTORY STORAGE
    sim_history = []

    # 2. GEOMETRY GENERATIONS
    sheet_nodes, sheet_edges, sheet_boundary, sheet_elems = sheet_mesh_generation(THICKNESS, LENGTH, RELATIVE_GROOVE_DEPTH, NOSE_RADIUS)
    punch_nodes, punch_edges = punch_mesh_generation( PUNCH_POSITION, PUNCH_ANGLE, PUNCH_RADIUS, PUNCH_HEIGHT, NUM_ELEM_PUNCH, PUNCH_REINFORCEMENT_MULTIPLE)
    die_nodes, die_edges = die_mesh_generation(DIE_ANGLE, DIE_GROOVE_LENGTH, DIE_CURVE_RADIUS, LENGTH, NUM_ELEM_DIE, DIE_REINFORCEMENT_MULTIPLE)
    grid = SpatialHash2D(COORDINATE_MIN, COORDINATE_MAX, GRID_SIZE)

    # 3. GLOAL VARIABLES
    Total_dofs = sheet_nodes.shape[1] * 2
    Disp_global = np.zeros(Total_dofs)
    t = 0
    fixed_dofs = np.argsort(np.abs(sheet_nodes[0]))[:5] * 2 # 5 most centered nodes
    sim_history.append({
        'time': 0.0,
        'sheet': sheet_nodes.copy(),
        'punch': punch_nodes.copy()
    })

    # 4. TIME LOOP
    for time_step_idx in range(NUM_LOOPS):
        punch_nodes += PUNCH_VELOCITY * TIME_STEP 
        t += TIME_STEP
        print(f"--- Step {time_step_idx}, Time {t * 1e3:.4f} ms ---")

        # 5. NEWTON-RAPHSON LOOP
        for iter in range(MAX_NEWTON_ITERATIONS):
            delta_u, res_norm = assemble_and_solve(
                sheet_nodes, sheet_elems, sheet_boundary,
                punch_nodes, punch_edges, die_nodes, die_edges,
                Disp_global, grid, fixed_dofs
            )

            Disp_global += delta_u

            print(f"   Iter {iter}: ||R|| = {res_norm:.4e}")
            if res_norm < NEWTON_RAPHSON_TOLERANCE:
                print("   >> Converged")
                break

        current_sheet_nodes = sheet_nodes + Disp_global.reshape((2, -1), order='F')
        sim_history.append({
            'time': t,
            'sheet': current_sheet_nodes.copy(),
            'punch': punch_nodes.copy()
        })

    # 6. POST-PROCESSING
    print("Simulation complete. Processing animation...")
    save_animation(
        sim_history, 
        die_nodes,
        sheet_edges, punch_edges, die_edges,
        COORDINATE_MIN, COORDINATE_MAX,
        filename="simulation_result.gif"
    )
    visualize(current_sheet_nodes, sheet_edges, punch_nodes, punch_edges, die_nodes, die_edges)
#%%
