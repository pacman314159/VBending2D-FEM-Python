import numpy as np

# Material Properties
YOUNG_MODULUS = 213 #GPa
POISSON_RATIO = 0.3
DENSITY = 7850e-9 #kg.mm^-3
YIELD_STRESS = 348 #MPa

GRAVITY_ACCEL = np.array([0, -9810]) #mm.s^-2

NUM_GP_CONTACT = 3
NUM_GP_2D_ELEM_PER_SIDE = 2

# Kinematics Constants
PUNCH_VELOCITY = np.array([0, -10]).reshape((2,1)) # mm/s
FRICTION_COEFF = 0.1
PENALTY_STIFFNESS_NORMAL = 1e6 # N/m^3
PENALTY_STIFFNESS_TANGENT = 1e6 # N/m^3

# For Plane Strain Problem
D_MATRIX = np.array([
    [1 - POISSON_RATIO, -POISSON_RATIO, 0],
    [-POISSON_RATIO, 1-POISSON_RATIO, 0],
    [0, 0, 0.5 * (1 - 2 * POISSON_RATIO)]
]) * YOUNG_MODULUS / (1 + POISSON_RATIO) / (1 - 2 * POISSON_RATIO)

MAX_NEWTON_ITERATIONS = 5

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
    if kwargs == {}: return np.zeros((8, 8))

    J = kwargs["J"]              # 2×2 Jacobian (undeformed)
    dN_dxi = kwargs["der"]       # 2×4 (dN/dxi, dN/deta)
    inv_J = np.linalg.inv(J)
    dN_dx = inv_J @ dN_dxi

    first_row = np.array([dN_dx[0, 0], 0, dN_dx[0, 1], 0, dN_dx[0, 2], 0, dN_dx[0, 3], 0])
    second_row = np.array([0, dN_dx[1, 0], 0, dN_dx[1, 1], 0, dN_dx[1, 2], 0, dN_dx[1, 3]])
    third_row = np.array([dN_dx[1, 0], dN_dx[0, 0], dN_dx[1, 1], dN_dx[0, 1], dN_dx[1, 2], dN_dx[0, 2], dN_dx[1, 3], dN_dx[0, 3]])
    B_linear = np.vstack((first_row, second_row, third_row))

    return B_linear.T @ D_MATRIX @ B_linear
#%%

#%%
def internal_force_func(xi, eta, **kwargs):
    """
    Total-Lagrangian internal force at a Gauss point for a 4-node quad.
    Returns 8x1 vector f_int_gp (this is the integrand: B^T * PK2).
    kwargs contains:
      - 'J' : Jacobian (2x2) of reference element (undeformed)
      - 'der': dN_dxi (2x4)
      - 'u_e': element nodal displacement vector (8,)  (used for computing strain via Green-Lagrange)
    """
    if kwargs == {}: return np.zeros(8)

    J = kwargs["J"]
    dN_dxi = kwargs["der"]   # (2,4) : dN/dxi, dN/deta
    u_e = kwargs["u_e"]      # length 8

    # Derivatives wrt reference coords (x0)
    invJ = np.linalg.inv(J)
    dN_dx = invJ @ dN_dxi     # (2,4)

    # build standard linear B (3 x 8) for Green-Lagrange linear part
    B_nonlinear = np.zeros((3, 8))
    for i in range(4):
        dNdx = dN_dx[0, i]
        dNdy = dN_dx[1, i]
        B_nonlinear[0, 2*i    ] = dNdx
        B_nonlinear[1, 2*i+1  ] = dNdy
        B_nonlinear[2, 2*i    ] = dNdy
        B_nonlinear[2, 2*i+1  ] = dNdx

    # compute deformation gradient and Green-Lagrange strain as in your code
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

    PK2_stress = D_MATRIX @ strain_vec

    f_int_gp = B_nonlinear.T @ PK2_stress   # (8,)
    return f_int_gp
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
