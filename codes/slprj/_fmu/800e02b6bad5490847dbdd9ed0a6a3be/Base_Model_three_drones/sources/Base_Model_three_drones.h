#ifndef Base_Model_three_drones_h_
#define Base_Model_three_drones_h_
#ifndef Base_Model_three_drones_COMMON_INCLUDES_
#define Base_Model_three_drones_COMMON_INCLUDES_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#include "rt_nonfinite.h"
#include "math.h"
#endif

#include "Base_Model_three_drones_types.h"
#include <string.h>
#ifndef rtmGetErrorStatus
#define rtmGetErrorStatus(rtm)         ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
#define rtmSetErrorStatus(rtm, val)    ((rtm)->errorStatus = (val))
#endif

#include "RTWCG_util_functions.h"

typedef struct {
  real_T DiscreteTimeIntegrator1_DSTATE[3];
  real_T DiscreteTimeIntegrator_DSTATE[3];
  real_T DiscreteTimeIntegrator2_DSTATE[9];
  real_T DiscreteTimeIntegrator3_DSTATE[3];
  real_T DiscreteTimeIntegrator1_DSTATE_f[3];
  real_T DiscreteTimeIntegrator1_DSTATE_g[3];
  real_T DiscreteTimeIntegrator1_DSTATE_l[3];
  real_T UnitDelay_DSTATE[3];
  real_T UnitDelay_DSTATE_b[3];
  real_T UnitDelay_DSTATE_k[3];
  real_T Product_DWORK4[9];
} DW_Base_Model_three_drones_T;

typedef struct {
  real_T Desired_Cable_Forces[9];
  real_T Desired_Cable_Forces_Derivatives[9];
} ExtU_Base_Model_three_drones_T;

typedef struct {
  real_T Load_Position[3];
  real_T Load_LinVelocity[3];
  real_T Load_Orientation[9];
  real_T Load_Orientation_Derivative[9];
  real_T Load_AngVelocity[3];
  real_T Drone_Positions[9];
  real_T Drones_LinVelocity[9];
} ExtY_Base_Model_three_drones_T;

struct P_Base_Model_three_drones_T_ {
  real_T DiscreteTimeIntegrator1_gainval;
  real_T DiscreteTimeIntegrator1_IC[3];
  real_T DiscreteTimeIntegrator_gainval;
  real_T DiscreteTimeIntegrator_IC;
  real_T DiscreteTimeIntegrator2_gainval;
  real_T DiscreteTimeIntegrator2_IC[9];
  real_T Constant_Value;
  real_T DiscreteTimeIntegrator3_gainval;
  real_T DiscreteTimeIntegrator3_IC;
  real_T Gain_Gain;
  real_T Gain1_Gain;
  real_T Gain2_Gain;
  real_T DiscreteTimeIntegrator1_gainval_g;
  real_T DiscreteTimeIntegrator1_IC_h[3];
  real_T DiscreteTimeIntegrator1_gainval_n;
  real_T DiscreteTimeIntegrator1_IC_e[3];
  real_T DiscreteTimeIntegrator1_gainval_i;
  real_T DiscreteTimeIntegrator1_IC_n[3];
  real_T UnitDelay_InitialCondition;
  real_T UnitDelay_InitialCondition_b;
  real_T UnitDelay_InitialCondition_o;
  real_T Constant_Value_m[9];
  real_T Saturation_UpperSat;
  real_T Saturation_LowerSat;
  real_T Constant_Value_g;
  real_T Gain_Gain_n;
  real_T Gain1_Gain_d;
  real_T Gain2_Gain_o;
  real_T Constant1_Value[9];
  real_T Saturation1_UpperSat;
  real_T Saturation1_LowerSat;
  real_T Constant2_Value[9];
  real_T Saturation2_UpperSat;
  real_T Saturation2_LowerSat;
  real_T Constant_Value_m4;
  real_T Gain_Gain_i;
  real_T Gain1_Gain_dl;
  real_T Gain2_Gain_e;
  real_T Constant_Value_f;
  real_T Gain_Gain_g;
  real_T Gain1_Gain_g;
  real_T Gain2_Gain_a;
  real_T Constant_Value_i;
  real_T Gain_Gain_c;
  real_T Gain1_Gain_f;
  real_T Gain2_Gain_m;
  real_T Constant_Value_fy[3];
  real_T Gain1_Gain_fl;
  boolean_T Assertion_Enabled;
};

struct tag_RTM_Base_Model_three_drones_T {
  const char_T *errorStatus;
};

extern P_Base_Model_three_drones_T Base_Model_three_drones_P;
extern DW_Base_Model_three_drones_T Base_Model_three_drones_DW;
extern ExtU_Base_Model_three_drones_T Base_Model_three_drones_U;
extern ExtY_Base_Model_three_drones_T Base_Model_three_drones_Y;
extern real_T Attachment_Point_Vectors[9];
extern real_T Cable_Resting_Length;
extern real_T Load_Damping_Coef;
extern real_T Load_Inertia_Matrix[9];
extern real_T Load_Mass;
extern real_T g;
extern RT_MODEL_Base_Model_three_drones_T *Base_Model_three_drones(void);
extern void Base_Model_three_drones_initialize(void);
extern void Base_Model_three_drones_step(void);
extern void Base_Model_three_drones_terminate(void);
extern RT_MODEL_Base_Model_three_drones_T *const Base_Model_three_drones_M;
extern void fmu_LogOutput(void);

#endif
