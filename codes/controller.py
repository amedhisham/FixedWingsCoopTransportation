import numpy as np

def get_reference_trajectory(t):
    """
    Calculates a straight-line trajectory in the X-direction.
    """
    v_x = 0.5      # Constant velocity in X (m/s)
    z_hover = 1.0  # Constant flight altitude (m)
    
    # 1. Desired Position
    p_Ld = np.array([
        # v_x * t,  # Moving forward in X
        0.0,
        0.0,      # Y is constant
        z_hover   # Z is constant
    ])
    
    # 2. Desired Linear Velocity
    v_Ld = np.array([
        # v_x,
        0.0,
        0.0,
        0.0
    ])
    
    # 3. Desired Orientation Matrix (Assuming no rotation, perfectly level)
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

    ew = ref_angVel - curr_angVel

    return ep , eR , ev , ew


def wrench_controller(ep,eR,ev,ew,curr_angVel,Load_Inertia_Matrix,Load_Mass,step_size):

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

    f_L_d = Load_Mass * g * e3 - (Kp @ ep) - (Kv @ ev) - (Ki @ wrench_controller.intg_ep)
    
    gyroscopic_term = np.cross(curr_angVel, Load_Inertia_Matrix @ curr_angVel)
    tau_L_d = gyroscopic_term - (KR @ eR) - (Kw @ ew) - (KiR @ wrench_controller.intg_eR)
    
    w_d = np.concatenate((f_L_d, tau_L_d))
    
    return w_d