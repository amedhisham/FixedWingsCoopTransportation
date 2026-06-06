import numpy as np
import casadi as ca
from scipy.linalg import null_space

def skew(v):
    """
    Returns the 3x3 skew-symmetric matrix of a vector.
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

def compute_structured_nullspace(curr_orientation_matrix, Attachment_Point_Vectors, n_carriers):
    
    # Step 1: Edge vectors for Hamiltonian cycle k -> k+1 -> ... -> 0
    edge_vectors = []
    for k in range(n_carriers):
        i = k
        j = (k + 1) % n_carriers  # wraps around: last point connects back to 0
        b_ij = curr_orientation_matrix @ (Attachment_Point_Vectors[j] - Attachment_Point_Vectors[i])
        # b_ij = np.eye(3)  @ (Attachment_Point_Vectors[j] - Attachment_Point_Vectors[i]) #static controller

        edge_vectors.append(b_ij)
    
    # Step 2: Incidence matrix H (n x n)
    # +1 at tail node, -1 at head node for each edge
    H = np.zeros((n_carriers, n_carriers))
    for k in range(n_carriers):
        i = k                        # tail: +1
        j = (k + 1) % n_carriers    # head: -1
        H[i, k] =  1
        H[j, k] = -1
    
    # Step 3: Block diagonal matrix of edge vectors (3n x n)
    D = np.zeros((3 * n_carriers, n_carriers))
    for k in range(n_carriers):
        D[3*k:3*(k+1), k] = edge_vectors[k]
    
    # Step 4: N = (H ⊗ I3) @ D  →  (3n x 3n) @ (3n x n) = (3n x n)
    N = np.kron(H, np.eye(3)) @ D
    
    return N

def calculate_grasp_and_nullspace(curr_orientation_matrix, Attachment_Point_Vectors,n_carriers):
    """
    Computes the Grasp Matrix (G), its pseudo-inverse, and Nullspace (N).
    
    Inputs:
        curr_orientation_matrix: 3x3 current rotation matrix of the load
        Attachment_Point_Vectors: List or array of 3D attachment points in the load frame F_L
             Shape should be (n_carriers, 3)
             
    Returns:
        G: 6 x 3n Grasp Matrix (Equation 8)
        G_pinv: 3n x 6 Right Pseudo-inverse of G
        N: 3n x (3n-6) Nullspace Basis Matrix
    """
    
    # Initialize the top and bottom halves of the Grasp matrix
    G_top = np.zeros((3, 3 * n_carriers))
    G_bottom = np.zeros((3, 3 * n_carriers))
    
    for i in range(n_carriers):
        # 1. Force block (I_3) maps cables to translational force
        G_top[:, i*3 : (i+1)*3] = np.eye(3)
        
        # 2. Torque block maps cables to moment about CoM
        # Equation 8: S(^B b_i) * R_L(t)^T
        Bb_i = Attachment_Point_Vectors[i,:]
        skew_Bb_i = skew(Bb_i)
        G_bottom[:, i*3 : (i+1)*3] = skew_Bb_i @ curr_orientation_matrix.T
        # G_bottom[:, i*3 : (i+1)*3] = skew_Bb_i @  np.eye(3).T #static controller 
        
    # Combine top and bottom blocks into the final 6 x 3n matrix
    G = np.vstack((G_top, G_bottom))
    
    # Calculate the right pseudo-inverse (G^dagger)
    G_pinv = np.linalg.pinv(G, rcond=1e-6)

    # Calculate the nullspace basis matrix N
    # Since rank(G) = 6, N will have dimensions 3n x (3n-6)
    N = compute_structured_nullspace(curr_orientation_matrix, Attachment_Point_Vectors, n_carriers)

    # print("G @ N =", np.round(G @ N, 6))
    
    return G, G_pinv, N

def cable_force_calculation(curr_orientation_matrix,Attachment_Point_Vectors,w_d,lambda_star,n_carriers):

   _ , Grasp_pinv_Matrix , Nullspace_Matrix = calculate_grasp_and_nullspace(curr_orientation_matrix,Attachment_Point_Vectors,n_carriers)

   desired_forces = Grasp_pinv_Matrix @ w_d + Nullspace_Matrix @ lambda_star
     
#    desired_forces = Grasp_pinv_Matrix @ w_d 


   return desired_forces , Grasp_pinv_Matrix

def init_optimizer(Cable_Resting_Length,n_carriers,w_pos, w_vel,phases):

    n_lambda = n_carriers

    # --- 2. SYMBOLIC VARIABLES & PARAMETERS ---
    opt_x = ca.SX.sym('x', 2) # [xi, A] 
    xi, A = opt_x[0], opt_x[1]

    # Data injected dynamically from the FMU at each step
    t_sym = ca.SX.sym('t')
    prev_x_sym = ca.SX.sym('prev_x', 2)
    prev_lam_sym = ca.SX.sym('prev_lam', n_lambda)
    prev_lam_dot_sym = ca.SX.sym('prev_lam_dot', n_lambda)

    w_d_sym = ca.SX.sym('w_d', 6)
    G_pinv_sym = ca.SX.sym('G_pinv', 12, 6)
    N_sym = ca.SX.sym('N', 12, n_lambda)         # Our 12x4 Nullspace slice
    N_dot_sym = ca.SX.sym('N_dot', 12, n_lambda) # Derivative of Nullspace
    e_sym = ca.SX.sym('e', 12)                   # External wrench derivatives
    v_L_sym = ca.SX.sym('v_L', 12)               # Carrier base velocities

    # --- 3. INTERNAL LAMBDA FUNCTIONS (Eq 23) ---
    lam = ca.SX.zeros(n_lambda)
    lam_dot = ca.SX.zeros(n_lambda)
    for j in range(n_lambda):
        lam[j] = A * ca.cos(xi * t_sym + phases[j])
        lam_dot[j] = -A * xi * ca.sin(xi * t_sym + phases[j])

    # --- 4. COST FUNCTION (Eq 20) ---
    cost = (xi - prev_x_sym[0])**2 + (A - prev_x_sym[1])**2 \
        + w_pos * ca.sumsqr(lam - prev_lam_sym) \
        + w_vel * ca.sumsqr(lam_dot - prev_lam_dot_sym)

    # --- 5. NON-STOPPING CONSTRAINTS (Eq 24) ---
    constraints = []

    # Total forces: f = G_pinv * w_d + N * lambda
    f_total = ca.mtimes(G_pinv_sym, w_d_sym) + ca.mtimes(N_sym, lam)

    for i in range(n_carriers):
        idx = slice(i*3, (i+1)*3)
        f_i = f_total[idx]
        
        # Tension calculation (Eq 16 magnitude)
        T_i_sq = ca.sumsqr(f_i) + 1e-6 # 1e-6 prevents divide-by-zero during solver init
        T_i = ca.sqrt(T_i_sq)
        
        # Projector matrix (Eq 18)
        q_i = f_i / T_i
        Pi_i = ca.SX.eye(3) - ca.mtimes(q_i, q_i.T)
        
        # Internal derivative term (Eq 21)
        g_i = ca.mtimes(N_dot_sym[idx, :], lam) + ca.mtimes(N_sym[idx, :], lam_dot)
        
        # Carrier velocity (Eq 22)
        v_Ri = v_L_sym[idx] + (Cable_Resting_Length / T_i) * ca.mtimes(Pi_i, e_sym[idx] + g_i)
        
        # Constraint: ||v_Ri||^2 >= epsilon^2
        constraints.append(ca.sumsqr(v_Ri))

    # --- 6. COMPILE SOLVER ---
    # Stack all parameters so we can feed them as one 1D array later
    p_sym = ca.vertcat(t_sym, prev_x_sym, prev_lam_sym, prev_lam_dot_sym, 
                    w_d_sym, ca.vec(G_pinv_sym), ca.vec(N_sym), ca.vec(N_dot_sym), 
                    e_sym, v_L_sym)

    nlp = {'x': opt_x, 'f': cost, 'g': ca.vertcat(*constraints), 'p': p_sym}
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.warm_start_init_point': 'yes'}
    casadi_solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    return casadi_solver


def optimizer(casadi_solver,time,curr_orientation_matrix,curr_linVel,curr_angVel,w_d,Attachment_Point_Vectors,epsilon,step_size,n_carriers,phases,bypass):


    if not hasattr(optimizer, "prev_x"):
        # Histories for Optimization
        xi0, A0 = 2.0, 1.2
        optimizer.prev_x = np.array([xi0, A0])
        optimizer.prev_x = np.array([2.0, 1.2]) 
        optimizer.prev_lam =  A0 * np.cos(xi0 * 0.0 + phases)
        optimizer.prev_lam_dot = -A0 * xi0 * np.sin(xi0 * 0.0 + phases)

        # Histories for derivatives
        optimizer.prev_w_d = w_d
        _, G_pinv_init, N_init = calculate_grasp_and_nullspace(curr_orientation_matrix, Attachment_Point_Vectors, n_carriers)
        optimizer.prev_G_pinv = G_pinv_init
        optimizer.prev_N = N_init


    _ , Grasp_pinv_Matrix , Nullspace_Matrix = calculate_grasp_and_nullspace(curr_orientation_matrix,Attachment_Point_Vectors,n_carriers)

    N = Nullspace_Matrix
    
    # 1. Finite Differences for Derivatives
    G_pinv_dot = (Grasp_pinv_Matrix - optimizer.prev_G_pinv) / step_size
    N_dot = (N - optimizer.prev_N) / step_size
    w_d_dot = (w_d - optimizer.prev_w_d) / step_size
    
    # 2. Pre-calculate the external derivative term 'e'
    e_total = (G_pinv_dot @ w_d) + (Grasp_pinv_Matrix @ w_d_dot)
    # print(e_total)
    # Calculate base carrier velocities v_Li (Eq 22 base term)
    v_L_stack = []
    for i in range(n_carriers):
        b_i_world = curr_orientation_matrix @ Attachment_Point_Vectors[i, :]
        v_Li = curr_linVel + np.cross(curr_angVel, b_i_world)
        v_L_stack.extend(v_Li)
    v_L_stack = np.array(v_L_stack)
    
    # 3. Pack parameters for CasADi
    p_val = np.concatenate([
        [time], optimizer.prev_x, optimizer.prev_lam, optimizer.prev_lam_dot, 
        w_d, Grasp_pinv_Matrix.flatten('F'), N.flatten('F'), N_dot.flatten('F'), 
        e_total, v_L_stack
    ])
    
    lbg = [] # Lower bounds
    lbg = [epsilon**2] * n_carriers

    # 4. SOLVE
    if bypass:
        opt_xi = 2.0
        opt_A = 1.2
        opt_x = np.array([opt_xi, opt_A])

    else:
        res = casadi_solver(x0=optimizer.prev_x, p=p_val, lbg=lbg, ubg=[np.inf]*n_carriers)
        
        opt_x = np.array(res['x']).flatten()
        opt_xi, opt_A = opt_x[0], opt_x[1]
        # print(f"Optimal Ksi: {opt_xi}")
        # print(f"Optimal A: {opt_A}")


    # 5. Extract our 4 Lambda Stars and their analytical derivatives
    lambda_star = opt_A * np.cos(opt_xi * time + phases)
    lambda_star_dot = -opt_A * opt_xi * np.sin(opt_xi * time + phases)

    # Calculate f_dot using Equation 19
    f_dot = e_total + (N_dot @ lambda_star) + (N @ lambda_star_dot)

    optimizer.prev_x = opt_x
    optimizer.prev_lam = lambda_star
    optimizer.prev_lam_dot = -opt_A * opt_xi * np.sin(opt_xi * time + phases)
    optimizer.prev_w_d = w_d
    optimizer.prev_G_pinv = Grasp_pinv_Matrix
    optimizer.prev_N = N

    return lambda_star , f_dot