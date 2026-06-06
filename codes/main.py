import fmpy
import numpy as np
import matplotlib.pyplot as plt
from fmpy.fmi2 import FMU2Slave
from scipy.spatial.transform import Rotation as R

from controller import  error_calculation , wrench_controller
from optimizer import cable_force_calculation , init_optimizer , optimizer
import time as time_module
from scipy.io import savemat

fmu_filename = 'Base_Model_three_drones.fmu'

# ---------------------------------------------------------
# PHASE 1: Parse the Dictionary
# ---------------------------------------------------------
# Read the model description to get the FMI 'Value References' (memory addresses)
model_desc = fmpy.read_model_description(fmu_filename)

vrs = {}
for var in model_desc.modelVariables:
    vrs[var.name] = var.valueReference

for key in vrs.keys():
        print(key)
print("-----------------------------")

# ---------------------------------------------------------
# PHASE 2: Extraction and Instantiation
# ---------------------------------------------------------
unzipdir = fmpy.extract(fmu_filename)

fmu = FMU2Slave(guid=model_desc.guid,
                unzipDirectory=unzipdir,
                modelIdentifier=model_desc.coSimulation.modelIdentifier,
                instanceName='PayloadSim')

fmu.instantiate()

# ---------------------------------------------------------
# PHASE 3: Initialization (The Boundary Check)
# ---------------------------------------------------------
fmu.setupExperiment(startTime=0.0)
fmu.enterInitializationMode()

# Global parameters
n_carriers = 3
epsilon = 0.2        # Minimum velocity threshold
w_pos, w_vel = 1e-6, 1e-6 # Cost weights 
phases = np.array([0, np.pi/3, 2*np.pi/3])
bypass_optimizer = 0 #bypass optimizer for debugging , static controller
bypass_controller = 0 #bypass controller for debugging , static controller with feedback

# 1. Your matrix and flattening (same as before)
J_L_matrix = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]) 
flat_J_L = J_L_matrix.flatten('F').tolist()

# 2. Generate the exact list of 9 string names Simulink created
# This creates: ['Load_Inertia_Matrix[1,1]', 'Load_Inertia_Matrix[1,2]', ... 'Load_Inertia_Matrix[1,9]']
inertia_keys = [f'Load_Inertia_Matrix[1,{i}]' for i in range(1, 10)]

# 3. Look up the FMI memory addresses (Value References) for those 9 keys
inertia_vrs = [vrs[key] for key in inertia_keys]

# 4. Inject all 9 values into the 9 addresses simultaneously
fmu.setReal(inertia_vrs, flat_J_L)

fmu.exitInitializationMode()
# ---------------------------------------------------------
# PHASE 4: The Outer Control Loop
# ---------------------------------------------------------
time = 0.0
step_size = 0.01  # 100 Hz control loop
end_time = 15.0

time_history = []
z_position_history = []

drone1_VelNorm_history = []
drone2_VelNorm_history = []
drone3_VelNorm_history = []
drone4_VelNorm_history = []

force_history = []
force_derv_history = []

Fz = 0.7 * 9.81/3
# Fz = 0
desired_forces = [0.0, 0.0,Fz, 0.0, 0.0, Fz, 0.0, 0.0, Fz]


# 1. PRE-COMPUTE THE MEMORY ADDRESSES 
# Create the 12 string names: 'Desired_Cable_Forces[1,1]' to '[1,12]'
force_keys = [f'Desired_Cable_Forces[{i}]' for i in range(1, n_carriers*3 + 1)]
deriv_keys = [f'Desired_Cable_Forces_Derivatives[{i}]' for i in range(1, n_carriers*3 + 1)]

# Look up the 12 memory addresses for both
force_vrs = [vrs[key] for key in force_keys]
deriv_vrs = [vrs[key] for key in deriv_keys]

Load_Inertia_Matrix = fmu.getReal([vrs['Load_Inertia_Matrix[1,1]'], 
                       vrs['Load_Inertia_Matrix[1,2]'], 
                       vrs['Load_Inertia_Matrix[1,3]'],
                       vrs['Load_Inertia_Matrix[1,4]'],
                       vrs['Load_Inertia_Matrix[1,5]'],
                       vrs['Load_Inertia_Matrix[1,6]'],
                       vrs['Load_Inertia_Matrix[1,7]'],
                       vrs['Load_Inertia_Matrix[1,8]'],
                       vrs['Load_Inertia_Matrix[1,9]']])

Load_Inertia_Matrix = np.array(Load_Inertia_Matrix).reshape((3, 3), order='F')


Attachment_Point_Vectors = fmu.getReal([vrs['Attachment_Point_Vectors[1,1]'], 
                       vrs['Attachment_Point_Vectors[1,2]'], 
                       vrs['Attachment_Point_Vectors[1,3]'],
                       vrs['Attachment_Point_Vectors[1,4]'],
                       vrs['Attachment_Point_Vectors[1,5]'],
                       vrs['Attachment_Point_Vectors[1,6]'],
                       vrs['Attachment_Point_Vectors[1,7]'],
                       vrs['Attachment_Point_Vectors[1,8]'],
                       vrs['Attachment_Point_Vectors[1,9]']])

Attachment_Point_Vectors = np.array(Attachment_Point_Vectors).reshape((n_carriers, 3))


    
Load_Mass = fmu.getReal([vrs['Load_Mass']])[0]

Cable_Resting_Length = fmu.getReal([vrs['Cable_Resting_Length']])[0]   


casadi_solver = init_optimizer(Cable_Resting_Length,n_carriers,w_pos, w_vel,phases)


start = time_module.time()


while time < end_time:
    
    # A. READ STATES: Get the current payload dynamics from Simulink
    
    curr_pos = np.array(fmu.getReal([vrs['Load_Position[1]'], 
                       vrs['Load_Position[2]'], 
                       vrs['Load_Position[3]']]))


    curr_linVel = np.array(fmu.getReal([vrs['Load_LinVelocity[1]'], 
                       vrs['Load_LinVelocity[2]'], 
                       vrs['Load_LinVelocity[3]']]))
    
    drone1_pos = np.array(fmu.getReal([vrs['Drone_Positions[1]'], 
                       vrs['Drone_Positions[2]'], 
                       vrs['Drone_Positions[3]']]))
    
    drone2_pos = np.array(fmu.getReal([vrs['Drone_Positions[4]'], 
                       vrs['Drone_Positions[5]'], 
                       vrs['Drone_Positions[6]']]))
    
    drone3_pos = np.array(fmu.getReal([vrs['Drone_Positions[7]'], 
                       vrs['Drone_Positions[8]'], 
                       vrs['Drone_Positions[9]']]))
    
    # drone4_pos = np.array(fmu.getReal([vrs['Drone_Positions[10]'], 
    #                    vrs['Drone_Positions[11]'], 
    #                    vrs['Drone_Positions[12]']]))
    
    drone1_linVel = np.array(fmu.getReal([vrs['Drones_LinVelocity[1]'], 
                       vrs['Drones_LinVelocity[2]'], 
                       vrs['Drones_LinVelocity[3]']]))
    
    drone2_linVel = np.array(fmu.getReal([vrs['Drones_LinVelocity[4]'], 
                       vrs['Drones_LinVelocity[5]'], 
                       vrs['Drones_LinVelocity[6]']]))
    
    drone3_linVel = np.array(fmu.getReal([vrs['Drones_LinVelocity[7]'], 
                       vrs['Drones_LinVelocity[8]'], 
                       vrs['Drones_LinVelocity[9]']]))
    
    # drone4_linVel = np.array(fmu.getReal([vrs['Drones_LinVelocity[10]'], 
    #                    vrs['Drones_LinVelocity[11]'], 
    #                    vrs['Drones_LinVelocity[12]']]))

    curr_orientation_matrix = fmu.getReal([vrs['Load_Orientation[1,1]'], 
                       vrs['Load_Orientation[1,2]'], 
                       vrs['Load_Orientation[1,3]'],
                       vrs['Load_Orientation[2,1]'],
                       vrs['Load_Orientation[2,2]'],
                       vrs['Load_Orientation[2,3]'],
                       vrs['Load_Orientation[3,1]'],
                       vrs['Load_Orientation[3,2]'],
                       vrs['Load_Orientation[3,3]']])
    
    curr_orientation_matrix = np.array(curr_orientation_matrix).reshape((3, 3), order='C')
   
    curr_orientation_matrix = np.round(curr_orientation_matrix, decimals=6)



    curr_angVel = np.array(fmu.getReal([vrs['Load_AngVelocity[1,1]'], 
                       vrs['Load_AngVelocity[2,1]'], 
                       vrs['Load_AngVelocity[3,1]']]))

    # Store the data
    time_history.append(time)
    current_pos_z = curr_pos[1]
    # print(currentPos_z)
    z_position_history.append(current_pos_z)

    drone1_VelNorm = np.linalg.norm(drone1_linVel)
    drone1_VelNorm_history.append(drone1_VelNorm)

    drone2_VelNorm = np.linalg.norm(drone2_linVel)
    drone2_VelNorm_history.append(drone2_VelNorm)

    drone3_VelNorm = np.linalg.norm(drone3_linVel)
    drone3_VelNorm_history.append(drone3_VelNorm)

    # drone4_VelNorm = np.linalg.norm(drone4_linVel)
    # drone4_VelNorm_history.append(drone4_VelNorm)

    # B. DESIRED WRENCH CALCULATION: Calculate error then pass it to wrench controller 
    ep , eR , ev , ew = error_calculation(curr_pos,curr_linVel,curr_orientation_matrix,curr_angVel,time)
    
    w_d = wrench_controller(ep,eR,ev,ew,curr_angVel,Load_Inertia_Matrix,Load_Mass,Attachment_Point_Vectors,step_size,n_carriers,bypass_controller,desired_forces)

    # drone1_VelNorm = np.linalg.norm(lambda_star)
    # drone1_VelNorm_history.append(drone1_VelNorm)


    # C. RUN OPTIMIZER: Pass states to fixed-wing optimizer
    lambda_star, f_dot = optimizer(casadi_solver,time,curr_orientation_matrix,curr_linVel,curr_angVel,w_d,Attachment_Point_Vectors,epsilon,step_size,n_carriers,phases,bypass_optimizer)
    
    # drone1_VelNorm = np.linalg.norm(drone1_pos)
    # drone1_VelNorm_history.append(w_d[2])

    desired_forces , Grasp_pinv_Matrix = cable_force_calculation(curr_orientation_matrix,Attachment_Point_Vectors,w_d,lambda_star,n_carriers)
    desired_forces = desired_forces.tolist()
    desired_force_derivatives = f_dot.tolist()
    # Fz = 0.7 * 9.81/4
    # desired_forces = [0.0, 0.0,Fz, 0.0, 0.0, Fz, 0.0, 0.0, Fz]
    # desired_force_derivatives = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # drone1_VelNorm = np.linalg.norm(drone1_linVel)
    # drone1_VelNorm_history.append(drone1_VelNorm)

    force_history.append(desired_forces)
    force_derv_history.append(desired_force_derivatives)

    # D. INJECT INPUTS: Send the 12x1 flat force vector into Inports
    fmu.setReal(force_vrs, desired_forces)
    fmu.setReal(deriv_vrs, desired_force_derivatives)

    # E. STEP SIMULATION: Tell the C++ solver to calculate the next 0.01 seconds
    # print(time)
    fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

    # F. ADVANCE CLOCK
    time += step_size

# ---------------------------------------------------------
# PHASE 5: Teardown
# ---------------------------------------------------------
elapsed = time_module.time() - start

fmu.terminate()
fmu.freeInstance()

# ---------------------------------------------------------
# PHASE 6: Plotting the Results
# ---------------------------------------------------------

data = {
    "t": time_history,
    "force": force_history,
    "force_derv" : force_derv_history
}

savemat("timeseries.mat", data)

print(f"Time : {elapsed:.5f} ")

# z_position_history = [abs(x) for x in z_position_history]

print(min(z_position_history))
plt.figure(figsize=(10, 5))
plt.step(time_history, z_position_history, label="FMU Payload Z", color='blue', where='post')
plt.title("Payload Z-Position over Time")
plt.xlabel("Time (s)")
plt.ylabel("Z Position (m)")
plt.grid(True)
plt.legend()
plt.show()

plt.plot(time_history, drone1_VelNorm_history, label='norm 1')
plt.plot(time_history, drone2_VelNorm_history, label='norm 2')
plt.plot(time_history, drone3_VelNorm_history, label='norm 3')
# plt.plot(time_history, drone4_VelNorm_history, label='norm 4')

plt.xlabel('Time')
plt.ylabel('Norm value')
plt.title('Norms over time')

plt.legend()
plt.grid(True)

plt.show()