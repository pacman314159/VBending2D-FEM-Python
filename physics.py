import numpy as np

# Material Properties
YOUNG_MODULUS = 213 #GPa
POISSON_RATIO = 0.3
DENSITY = 7.85e-9 #kg.mm^-3
YIELD_STRESS = 348 #MPa

GRAVITY_ACCEL = np.array([0, -9810]) #mm.s^-2

NUM_GP_CONTACT = 3
NUM_GP_2D_ELEM_PER_SIDE = 2

# Kinematics Constants
PUNCH_VELOCITY = np.array([0, -10]).reshape((2,1)) # mm/s
FRICTION_COEFF = 0.1
PENALTY_STIFFNESS_NORMAL = 1e5

# For Plane Strain Problem
D_MATRIX = np.array([
    [1 - POISSON_RATIO, -POISSON_RATIO, 0],
    [-POISSON_RATIO, 1-POISSON_RATIO, 0],
    [0, 0, 0.5 * (1 - 2 * POISSON_RATIO)]
]) * YOUNG_MODULUS / (1 + POISSON_RATIO) / (1 - 2 * POISSON_RATIO)

MAX_NEWTON_ITERATIONS = 15
TIME_STEP = 0.5e-3 #s
NEWTON_RAPHSON_TOLERANCE = 1e1

#%%
def body_force_func(xi, eta, **kwargs):
    if kwargs == {}: return np.zeros(8)

    N1 = 0.25 * (1 - xi) * (1 - eta)
    N2 = 0.25 * (1+ xi) * (1 - eta)
    N3 = 0.25 * (1 + xi) * (1 + eta)
    N4 = 0.25 * (1 - xi) * (1 + eta)
    N = np.array([
        [N1, 0, N2, 0, N3, 0, N4, 0],
        [0, N1, 0, N2, 0, N3, 0, N4]
    ])
    return N.T @ GRAVITY_ACCEL * DENSITY
#%%

#%%
def material_stiffness_matrix_func(xi, eta, **kwargs):
    """
    Computes the Material Tangent Stiffness Matrix (K_L) for Total Lagrangian.
    K_L = Integral( B_L.T * D * B_L ) dV0
    (Simplified for St. Venant-Kirchhoff)
    """
    if kwargs == {}: return np.zeros((8, 8))

    J = kwargs["J"]             # 2x2 Jacobian of UNDEFORMED element (Reference)
    dN_dxi = kwargs["der"]      # 2x4 Local derivatives
    u_e = kwargs["u_e"]         # 8x1 Nodal displacements

    inv_J = np.linalg.inv(J)
    dN_dX = inv_J @ dN_dxi

    u_nodes = u_e.reshape(4, 2)
    grad_u = u_nodes.T @ dN_dX.T 
    F = np.eye(2) + grad_u
    B_L = np.zeros((3, 8))
    
    F11, F12 = F[0, 0], F[0, 1]
    F21, F22 = F[1, 0], F[1, 1]

    for i in range(4):
        dN_dX_i = dN_dX[0, i]
        dN_dY_i = dN_dX[1, i]
        B_L[0, 2*i]     = F11 * dN_dX_i
        B_L[0, 2*i+1]   = F21 * dN_dX_i
        B_L[1, 2*i]     = F12 * dN_dY_i
        B_L[1, 2*i+1]   = F22 * dN_dY_i
        B_L[2, 2*i]     = F11 * dN_dY_i + F12 * dN_dX_i
        B_L[2, 2*i+1]   = F21 * dN_dY_i + F22 * dN_dX_i

    # 4. Stiffness Calculation
    return B_L.T @ D_MATRIX @ B_L
#%%

#%%
def internal_force_func(xi, eta, **kwargs):
    """
    Computes Internal Force Vector.
    F_int = Integral( B_L.T * S ) dV0
    where S is the Second Piola-Kirchhoff stress.
   
    """
    if kwargs == {}: return np.zeros(8)

    J = kwargs["J"]
    dN_dxi = kwargs["der"]
    u_e = kwargs["u_e"]

    # 1. Derivatives w.r.t Initial Coordinates X
    inv_J = np.linalg.inv(J)
    dN_dX = inv_J @ dN_dxi

    # 2. Deformation Gradient F
    u_nodes = u_e.reshape(4, 2)
    grad_u = u_nodes.T @ dN_dX.T
    F = np.eye(2) + grad_u

    # 3. Green-Lagrange Strain E = 0.5 * (F.T*F - I)
    C = F.T @ F
    E = 0.5 * (C - np.eye(2))
    
    # Voigt notation for 2D Plane Strain: [E_11, E_22, 2*E_12]
    strain_vec = np.array([E[0,0], E[1,1], 2.0*E[0,1]])

    PK2_stress = D_MATRIX @ strain_vec # Second Piola-Kirchhoff Stress

    # Construct B_L Matrix
    B_L = np.zeros((3, 8))
    F11, F12 = F[0, 0], F[0, 1]
    F21, F22 = F[1, 0], F[1, 1]

    for i in range(4):
        dN_dX_i = dN_dX[0, i]
        dN_dY_i = dN_dX[1, i]
        
        B_L[0, 2*i]     = F11 * dN_dX_i
        B_L[0, 2*i+1]   = F21 * dN_dX_i
        B_L[1, 2*i]     = F12 * dN_dY_i
        B_L[1, 2*i+1]   = F22 * dN_dY_i
        B_L[2, 2*i]     = F11 * dN_dY_i + F12 * dN_dX_i
        B_L[2, 2*i+1]   = F21 * dN_dY_i + F22 * dN_dX_i

    return B_L.T @ PK2_stress
#%%

#%%
def geometric_matrix_func(xi, eta, **kwargs):
    """
    Return 8x8 geometric stiffness matrix contribution at this Gauss point (the integrand).
    Assumes kwargs contains: 'J', 'der', 'u_e'.
    The integrator will multiply by detJ * weight * WIDTH later.
    Implementation is:
      K_geo_gp[a,b] = dN_dx[i,a] * T[i,j] * dN_dx[j,b]  (expanded into DOFs)
    where T is the 2x2 Second Piola-Kirchhoff stress matrix.
    """
    if kwargs == {}: return np.zeros((8,8))

    J = kwargs["J"]
    dN_dxi = kwargs["der"]
    u_e = kwargs["u_e"]

    invJ = np.linalg.inv(J)
    dN_dx = invJ @ dN_dxi   # (2,4)

    # Compute PK2 stress at GP (same as in internal force)
    u_nodes = u_e.reshape(4,2)
    F = np.eye(2)
    for i in range(4):
        dNix = dN_dx[0, i]; dNiy = dN_dx[1, i]
        F[0,0] += u_nodes[i,0] * dNix
        F[0,1] += u_nodes[i,0] * dNiy
        F[1,0] += u_nodes[i,1] * dNix
        F[1,1] += u_nodes[i,1] * dNiy

    C = F.T @ F
    E = 0.5 * (C - np.eye(2)) # Green-Lagrange strain tensor
    strain_vec = np.array([E[0,0], E[1,1], 2.0*E[0,1]])
    PK2 = D_MATRIX @ strain_vec

    # get 2x2 matrix form of PK2 (S)
    Sxx = PK2[0]
    Syy = PK2[1]
    Sxy = PK2[2] / 2.0
    S =  np.array([[Sxx, Sxy],
                   [Sxy, Syy]])

    K_geo = np.zeros((8,8))
    # assemble: K_ab^{ij} = dN_dx[i,a] * S[i,j] * dN_dx[j,b]
    for a in range(4):
        for b in range(4):
            for i in range(2):
                for j in range(2):
                    row = 2*a + i
                    col = 2*b + j
                    K_geo[row, col] += dN_dx[i, a] * S[i, j] * dN_dx[j, b]

    return K_geo
#%%

#%%
def apply_dirichlet(K, R, fixed_dofs):
    """
    Applies Dirichlet BCs in-place.
    """

    dofs = np.array(fixed_dofs)
    K[dofs, :] = 0.0
    K[:, dofs] = 0.0
    K[dofs, dofs] = 1.0
    R[dofs] = 0.0
    
    return K, R
#%%
