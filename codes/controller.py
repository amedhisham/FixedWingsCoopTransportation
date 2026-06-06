import numpy as np
from optimizer import skew


def get_reference_trajectory(t):
    """
    Calculates a piecewise straight-line trajectory:

    """
    v_move = 0.0   # Speed during the moving phase (m/s) - change this to whatever you want
    z_hover = 1.39  # Constant flight altitude (m)
    
    # Initialize variables
    x = 0.5
    y = 0.0
    vx = 0.0
    
    # 0. Piecewise Logic
    if t <= 5.0:
        # Phase 1: Hold position at the start
        x = 0.0
        y = 0.0
        vx = 0.0
        
    elif t <= 15.0:
        # Phase 2: Move forward. 
        # Note: We use (t - 5.0) so the position starts counting up from 0.0 exactly at t = 5.
        x = v_move * (t - 5.0)
        vx = v_move
        
    else:
        # Phase 3: Stop and lock final position.
        # The position is frozen at exactly where it ended at t = 15 (which is v_move * 10 seconds of travel)
        x = v_move * 10.0 
        vx = 0.0

    # 1. Desired Position
    p_Ld = np.array([
        x,        # Piecewise X position
        y,      # Y is constant
        z_hover   # Z is constant
    ])
    
    # 2. Desired Linear Velocity
    v_Ld = np.array([
        vx,       # Piecewise X velocity
        0.0,
        0.0
    ])
    
    # 3. Desired Orientation Matrix (Perfectly level, no rotation)
    R_Ld = np.eye(3) 
    
    # 4. Desired Angular Velocity (Zero rotation)
    omega_Ld = np.array([
        0.0,
        0.0,
        0.0
    ])
    
    return p_Ld, v_Ld, R_Ld, omega_Ld

def vee(S):
    """
    The vee operator (inverse of the skew-symmetric operator).
    Extracts the 3D vector from a 3x3 skew-symmetric matrix.
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def error_calculation(curr_pos,curr_linVel,curr_orientation_matrix,curr_angVel,time):

    ref_pos,ref_linVel,ref_orientation_matrix,ref_angVel = get_reference_trajectory(time)

    ep = curr_pos - ref_pos

    eR = 0.5 * vee(ref_orientation_matrix.T @ curr_orientation_matrix - curr_orientation_matrix.T @ ref_orientation_matrix)

    ev = curr_linVel - ref_linVel

    ew = curr_angVel - curr_orientation_matrix.T @ ref_orientation_matrix @ ref_angVel

    return ep , eR , ev , ew

def compute_structured_nullspace(curr_orientation_matrix, Attachment_Point_Vectors, n_carriers):
    
    # Step 1: Edge vectors for Hamiltonian cycle k -> k+1 -> ... -> 0
    edge_vectors = []
    for k in range(n_carriers):
        i = k
        j = (k + 1) % n_carriers  # wraps around: last point connects back to 0
        b_ij = curr_orientation_matrix @ (Attachment_Point_Vectors[j] - Attachment_Point_Vectors[i])
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

def wrench_controller(ep,eR,ev,ew,curr_angVel,Load_Inertia_Matrix,Load_Mass,Attachment_Point_Vectors,step_size,n_carriers,bypass,desired_forces):

    kp_scalar = 5.0
    kv_scalar = 2.0
    ki_scalar = 0.9
    kR_scalar = 0.5
    kw_scalar = 0.06
    kiR_scalar = 0.1

    Kp = kp_scalar * np.eye(3)
    Kv = kv_scalar * np.eye(3)
    Ki = ki_scalar * np.eye(3)
    KR = kR_scalar * np.eye(3)
    Kw = kw_scalar * np.eye(3)
    KiR = kiR_scalar * np.eye(3)

    g = 9.81
    e3 = np.array([0.0, 0.0, 1.0])

    if not hasattr(wrench_controller, "intg_ep"):
        wrench_controller.intg_ep = np.zeros(3)
        wrench_controller.intg_eR = np.zeros(3)
    
    wrench_controller.intg_ep += ep * step_size
    wrench_controller.intg_eR += eR * step_size
    if bypass:

        
        G_top = np.zeros((3, 3 * n_carriers))
        G_bottom = np.zeros((3, 3 * n_carriers))
        
        for i in range(n_carriers):
            # 1. Force block (I_3) maps cables to translational force
            G_top[:, i*3 : (i+1)*3] = np.eye(3)
            
            # 2. Torque block maps cables to moment about CoM
            # Equation 8: S(^B b_i) * R_L(t)^T
            Bb_i = Attachment_Point_Vectors[i,:]
            skew_Bb_i = skew(Bb_i)
            G_bottom[:, i*3 : (i+1)*3] = skew_Bb_i @ np.eye(3).T
            
        # Combine top and bottom blocks into the final 6 x 3n matrix
        G = np.vstack((G_top, G_bottom))

        desired_forces = np.array(desired_forces).reshape(n_carriers * 3, 1)

        w_d = G @ desired_forces
        w_d = w_d.flatten()
    else:
        f_L_d = Load_Mass * g * e3 - (Kp @ ep) - (Kv @ ev) - (Ki @ wrench_controller.intg_ep)
        
        gyroscopic_term = np.cross(curr_angVel, Load_Inertia_Matrix @ curr_angVel)
        tau_L_d = gyroscopic_term - (KR @ eR) - (Kw @ ew) - (KiR @ wrench_controller.intg_eR)
        
        w_d = np.concatenate((f_L_d, tau_L_d))
    
    return w_d