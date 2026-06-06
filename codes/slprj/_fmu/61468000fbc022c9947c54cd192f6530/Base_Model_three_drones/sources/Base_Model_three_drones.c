#include "Base_Model_three_drones_macros.h"
#include "Base_Model_three_drones.h"
#include <string.h>
#include <emmintrin.h>
#include <math.h>
#include "rt_assert.h"
#include "Base_Model_three_drones_private.h"
#include "rtwtypes.h"

real_T Attachment_Point_Vectors[9] = { 0.0, 0.66666666666666663, 0.0,
  -0.57735026918962584, -0.33333333333333331, 0.35, 0.57735026918962584,
  -0.33333333333333331, -0.35 } ;

real_T Cable_Resting_Length = 0.8;
real_T Load_Damping_Coef = 1.0;
real_T Load_Inertia_Matrix[9] = { 0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01
} ;

real_T Load_Mass = 0.7;
real_T g = 9.81;
DW_Base_Model_three_drones_T Base_Model_three_drones_DW;
ExtU_Base_Model_three_drones_T Base_Model_three_drones_U;
ExtY_Base_Model_three_drones_T Base_Model_three_drones_Y;
static RT_MODEL_Base_Model_three_drones_T Base_Model_three_drones_M_;
RT_MODEL_Base_Model_three_drones_T *const Base_Model_three_drones_M =
  &Base_Model_three_drones_M_;
void rt_invd3x3_snf(const real_T u[9], real_T y[9])
{
  __m128d tmp;
  __m128d tmp_1;
  real_T x[9];
  real_T tmp_0[2];
  real_T absx11;
  real_T absx21;
  real_T absx31;
  int32_T p1;
  int32_T p2;
  int32_T p3;
  memcpy(&x[0], &u[0], 9U * sizeof(real_T));
  p1 = 1;
  p2 = 3;
  p3 = 6;
  absx11 = fabs(u[0]);
  absx21 = fabs(u[1]);
  absx31 = fabs(u[2]);
  if ((absx21 > absx11) && (absx21 > absx31)) {
    p1 = 4;
    p2 = 0;
    x[0] = u[1];
    x[1] = u[0];
    x[3] = u[4];
    x[4] = u[3];
    x[6] = u[7];
    x[7] = u[6];
  } else if (absx31 > absx11) {
    p1 = 7;
    p3 = 0;
    x[2] = x[0];
    x[0] = u[2];
    x[5] = x[3];
    x[3] = u[5];
    x[8] = x[6];
    x[6] = u[8];
  }

  absx11 = x[1] / x[0];
  tmp = _mm_div_pd(_mm_loadu_pd(&x[1]), _mm_set1_pd(x[0]));
  _mm_storeu_pd(&tmp_0[0], tmp);
  x[1] = tmp_0[0];
  x[2] /= x[0];
  tmp = _mm_set_pd(tmp_0[1], absx11);
  tmp_1 = _mm_sub_pd(_mm_loadu_pd(&x[4]), _mm_mul_pd(tmp, _mm_set1_pd(x[3])));
  _mm_storeu_pd(&x[4], tmp_1);
  tmp = _mm_sub_pd(_mm_loadu_pd(&x[7]), _mm_mul_pd(tmp, _mm_set1_pd(x[6])));
  _mm_storeu_pd(&x[7], tmp);
  if (fabs(x[5]) > fabs(x[4])) {
    int32_T itmp;
    itmp = p2;
    p2 = p3;
    p3 = itmp;
    x[1] = tmp_0[1];
    x[2] = absx11;
    absx11 = x[4];
    x[4] = x[5];
    x[5] = absx11;
    absx11 = x[7];
    x[7] = x[8];
    x[8] = absx11;
  }

  absx11 = x[5] / x[4];
  x[5] = absx11;
  x[8] -= absx11 * x[7];
  absx11 = (x[1] * absx11 - x[2]) / x[8];
  absx21 = -(x[7] * absx11 + x[1]) / x[4];
  y[p1 - 1] = ((1.0 - x[3] * absx21) - x[6] * absx11) / x[0];
  y[p1] = absx21;
  y[p1 + 1] = absx11;
  absx11 = -x[5] / x[8];
  absx21 = (1.0 - x[7] * absx11) / x[4];
  y[p2] = -(x[3] * absx21 + x[6] * absx11) / x[0];
  y[p2 + 1] = absx21;
  y[p2 + 2] = absx11;
  absx11 = 1.0 / x[8];
  absx21 = -x[7] * absx11 / x[4];
  y[p3] = -(x[3] * absx21 + x[6] * absx11) / x[0];
  y[p3 + 1] = absx21;
  y[p3 + 2] = absx11;
}

void Base_Model_three_drones_step(void)
{
  __m128d tmp_4;
  __m128d tmp_5;
  __m128d tmp_6;
  __m128d tmp_8;
  __m128d tmp_9;
  real_T rtb_RL_dot[9];
  real_T rtb_Transpose[9];
  real_T tmp[9];
  real_T tmp_0[9];
  real_T tmp_1[9];
  real_T tmp_2[9];
  real_T rtb_Force_Sum[3];
  real_T rtb_MatrixMultiply5[3];
  real_T rtb_Product4[3];
  real_T rtb_Product5[3];
  real_T rtb_Product6[3];
  real_T rtb_Sum3[3];
  real_T rtb_Torque_Sum[3];
  real_T rtb_Torque_Sum_0[3];
  real_T rtb_Transpose1[3];
  real_T rtb_pH2_dot[3];
  real_T tmp_3[3];
  real_T tmp_7[2];
  real_T Desired_Cable_Forces;
  real_T Desired_Cable_Forces_Derivatives;
  real_T Desired_Cable_Forces_Derivatives_0;
  real_T Desired_Cable_Forces_Derivatives_1;
  real_T localProduct;
  real_T rtb_Gain;
  real_T rtb_RL_dot_0;
  real_T rtb_RL_dot_1;
  int32_T i;
  int32_T i_0;
  int32_T rtb_Transpose_tmp;
  int32_T rtb_Transpose_tmp_0;
  int32_T tmp_a;
  Base_Model_three_drones_Y.Load_Position[0] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE[0];
  Base_Model_three_drones_Y.Load_LinVelocity[0] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[0];
  Base_Model_three_drones_Y.Load_Position[1] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE[1];
  Base_Model_three_drones_Y.Load_LinVelocity[1] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[1];
  Base_Model_three_drones_Y.Load_Position[2] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE[2];
  Base_Model_three_drones_Y.Load_LinVelocity[2] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[2];
  memcpy(&Base_Model_three_drones_Y.Load_Orientation[0],
         &Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[0], 9U *
         sizeof(real_T));
  tmp[0] = Base_Model_three_drones_P.Constant_Value;
  tmp[3] = Base_Model_three_drones_P.Gain_Gain *
    Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[2];
  tmp[6] = Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[1];
  tmp[1] = Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[2];
  tmp[4] = Base_Model_three_drones_P.Constant_Value;
  tmp_9 = _mm_mul_pd(_mm_set_pd(Base_Model_three_drones_P.Gain2_Gain,
    Base_Model_three_drones_P.Gain1_Gain), _mm_loadu_pd
                     (&Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[
                      0]));
  _mm_storeu_pd(&tmp_7[0], tmp_9);
  tmp[7] = tmp_7[0];
  tmp[2] = tmp_7[1];
  tmp[5] = Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[0];
  tmp[8] = Base_Model_three_drones_P.Constant_Value;
  for (i = 0; i < 3; i++) {
    rtb_RL_dot_0 = 0.0;
    rtb_Gain = 0.0;
    rtb_RL_dot_1 = 0.0;
    for (i_0 = 0; i_0 < 3; i_0++) {
      Desired_Cable_Forces = tmp[3 * i + i_0];
      tmp_9 = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd
        (&Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[3 * i_0]),
        _mm_set1_pd(Desired_Cable_Forces)), _mm_set_pd(rtb_Gain, rtb_RL_dot_0));
      _mm_storeu_pd(&tmp_7[0], tmp_9);
      rtb_RL_dot_0 = tmp_7[0];
      rtb_Gain = tmp_7[1];
      rtb_RL_dot_1 += Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE
        [3 * i_0 + 2] * Desired_Cable_Forces;
    }

    rtb_RL_dot[3 * i + 2] = rtb_RL_dot_1;
    rtb_RL_dot[3 * i + 1] = rtb_Gain;
    rtb_RL_dot[3 * i] = rtb_RL_dot_0;
  }

  memcpy(&Base_Model_three_drones_Y.Load_Orientation_Derivative[0], &rtb_RL_dot
         [0], 9U * sizeof(real_T));
  Base_Model_three_drones_Y.Load_AngVelocity[0] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[0];
  Base_Model_three_drones_Y.Drone_Positions[0] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_f[0];
  Base_Model_three_drones_Y.Drone_Positions[3] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_g[0];
  Base_Model_three_drones_Y.Drone_Positions[6] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_l[0];
  Base_Model_three_drones_Y.Drones_LinVelocity[0] =
    Base_Model_three_drones_DW.UnitDelay_DSTATE[0];
  Base_Model_three_drones_Y.Drones_LinVelocity[3] =
    Base_Model_three_drones_DW.UnitDelay_DSTATE_b[0];
  Base_Model_three_drones_Y.Drones_LinVelocity[6] =
    Base_Model_three_drones_DW.UnitDelay_DSTATE_k[0];
  Base_Model_three_drones_Y.Load_AngVelocity[1] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[1];
  Base_Model_three_drones_Y.Drone_Positions[1] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_f[1];
  Base_Model_three_drones_Y.Drone_Positions[4] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_g[1];
  Base_Model_three_drones_Y.Drone_Positions[7] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_l[1];
  Base_Model_three_drones_Y.Drones_LinVelocity[1] =
    Base_Model_three_drones_DW.UnitDelay_DSTATE[1];
  Base_Model_three_drones_Y.Drones_LinVelocity[4] =
    Base_Model_three_drones_DW.UnitDelay_DSTATE_b[1];
  Base_Model_three_drones_Y.Drones_LinVelocity[7] =
    Base_Model_three_drones_DW.UnitDelay_DSTATE_k[1];
  Base_Model_three_drones_Y.Load_AngVelocity[2] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[2];
  Base_Model_three_drones_Y.Drone_Positions[2] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_f[2];
  Base_Model_three_drones_Y.Drone_Positions[5] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_g[2];
  Base_Model_three_drones_Y.Drone_Positions[8] =
    Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_l[2];
  Base_Model_three_drones_Y.Drones_LinVelocity[2] =
    Base_Model_three_drones_DW.UnitDelay_DSTATE[2];
  Base_Model_three_drones_Y.Drones_LinVelocity[5] =
    Base_Model_three_drones_DW.UnitDelay_DSTATE_b[2];
  Base_Model_three_drones_Y.Drones_LinVelocity[8] =
    Base_Model_three_drones_DW.UnitDelay_DSTATE_k[2];
  localProduct = sqrt((Base_Model_three_drones_U.Desired_Cable_Forces[0] *
                       Base_Model_three_drones_U.Desired_Cable_Forces[0] +
                       Base_Model_three_drones_U.Desired_Cable_Forces[1] *
                       Base_Model_three_drones_U.Desired_Cable_Forces[1]) +
                      Base_Model_three_drones_U.Desired_Cable_Forces[2] *
                      Base_Model_three_drones_U.Desired_Cable_Forces[2]);
  if (localProduct > Base_Model_three_drones_P.Saturation_UpperSat) {
    localProduct = Base_Model_three_drones_P.Saturation_UpperSat;
  } else if (localProduct < Base_Model_three_drones_P.Saturation_LowerSat) {
    localProduct = Base_Model_three_drones_P.Saturation_LowerSat;
  }

  tmp_9 = _mm_set1_pd(localProduct);
  _mm_storeu_pd(&rtb_Transpose1[0], _mm_div_pd(_mm_loadu_pd
    (&Base_Model_three_drones_U.Desired_Cable_Forces[0]), tmp_9));
  rtb_Transpose1[2] = Base_Model_three_drones_U.Desired_Cable_Forces[2] /
    localProduct;
  for (i = 0; i < 3; i++) {
    tmp_8 = _mm_sub_pd(_mm_loadu_pd(&Base_Model_three_drones_P.Constant_Value_m
      [3 * i]), _mm_mul_pd(_mm_loadu_pd(&rtb_Transpose1[0]), _mm_set1_pd
      (rtb_Transpose1[i])));
    _mm_storeu_pd(&tmp[3 * i], tmp_8);
    i_0 = 3 * i + 2;
    tmp[i_0] = Base_Model_three_drones_P.Constant_Value_m[i_0] - rtb_Transpose1
      [2] * rtb_Transpose1[i];
  }

  Desired_Cable_Forces_Derivatives =
    Base_Model_three_drones_U.Desired_Cable_Forces_Derivatives[1];
  Desired_Cable_Forces_Derivatives_0 =
    Base_Model_three_drones_U.Desired_Cable_Forces_Derivatives[0];
  Desired_Cable_Forces_Derivatives_1 =
    Base_Model_three_drones_U.Desired_Cable_Forces_Derivatives[2];
  for (i = 0; i <= 0; i += 2) {
    tmp_8 = _mm_loadu_pd(&tmp[i + 3]);
    tmp_5 = _mm_loadu_pd(&tmp[i]);
    tmp_6 = _mm_loadu_pd(&tmp[i + 6]);
    _mm_storeu_pd(&rtb_Product4[i], _mm_mul_pd(_mm_div_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_8, _mm_set1_pd(Desired_Cable_Forces_Derivatives)),
       _mm_mul_pd(tmp_5, _mm_set1_pd(Desired_Cable_Forces_Derivatives_0))),
      _mm_mul_pd(tmp_6, _mm_set1_pd(Desired_Cable_Forces_Derivatives_1))), tmp_9),
      _mm_set1_pd(Cable_Resting_Length)));
  }

  for (i = 2; i < 3; i++) {
    rtb_Product4[i] = ((tmp[i + 3] * Desired_Cable_Forces_Derivatives + tmp[i] *
                        Desired_Cable_Forces_Derivatives_0) + tmp[i + 6] *
                       Desired_Cable_Forces_Derivatives_1) / localProduct *
      Cable_Resting_Length;
  }

  rtb_Transpose[0] = Base_Model_three_drones_P.Constant_Value_g;
  rtb_Transpose[3] = Base_Model_three_drones_P.Gain_Gain_n *
    Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[2];
  rtb_Transpose[6] = Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[1];
  rtb_Transpose[1] = Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[2];
  rtb_Transpose[4] = Base_Model_three_drones_P.Constant_Value_g;
  tmp_9 = _mm_mul_pd(_mm_set_pd(Base_Model_three_drones_P.Gain2_Gain_o,
    Base_Model_three_drones_P.Gain1_Gain_d), _mm_loadu_pd
                     (&Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[
                      0]));
  _mm_storeu_pd(&tmp_7[0], tmp_9);
  rtb_Transpose[7] = tmp_7[0];
  rtb_Transpose[2] = tmp_7[1];
  rtb_Transpose[5] = Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[0];
  rtb_Transpose[8] = Base_Model_three_drones_P.Constant_Value_g;
  rtb_RL_dot_0 = 0.0;
  rtb_Gain = 0.0;
  rtb_RL_dot_1 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_9 = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Transpose[3 * i]),
      _mm_set1_pd(Attachment_Point_Vectors[i])), _mm_set_pd(rtb_Gain,
      rtb_RL_dot_0));
    _mm_storeu_pd(&tmp_7[0], tmp_9);
    rtb_RL_dot_0 = tmp_7[0];
    rtb_Gain = tmp_7[1];
    rtb_RL_dot_1 += rtb_Transpose[3 * i + 2] * Attachment_Point_Vectors[i];
  }

  Desired_Cable_Forces = 0.0;
  for (i = 0; i < 3; i++) {
    rtb_Transpose1[i] =
      ((Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[i + 3] *
        rtb_Gain + Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[i] *
        rtb_RL_dot_0) +
       Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[i + 6] *
       rtb_RL_dot_1) +
      Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[i];
    localProduct = Base_Model_three_drones_U.Desired_Cable_Forces[i + 3];
    Desired_Cable_Forces += localProduct * localProduct;
  }

  localProduct = sqrt(Desired_Cable_Forces);
  if (localProduct > Base_Model_three_drones_P.Saturation1_UpperSat) {
    localProduct = Base_Model_three_drones_P.Saturation1_UpperSat;
  } else if (localProduct < Base_Model_three_drones_P.Saturation1_LowerSat) {
    localProduct = Base_Model_three_drones_P.Saturation1_LowerSat;
  }

  _mm_storeu_pd(&rtb_Force_Sum[0], _mm_div_pd(_mm_loadu_pd
    (&Base_Model_three_drones_U.Desired_Cable_Forces[3]), _mm_set1_pd
    (localProduct)));
  rtb_Force_Sum[2] = Base_Model_three_drones_U.Desired_Cable_Forces[5] /
    localProduct;
  for (i = 0; i < 3; i++) {
    tmp_9 = _mm_sub_pd(_mm_loadu_pd(&Base_Model_three_drones_P.Constant1_Value[3
      * i]), _mm_mul_pd(_mm_loadu_pd(&rtb_Force_Sum[0]), _mm_set1_pd
                        (rtb_Force_Sum[i])));
    _mm_storeu_pd(&tmp[3 * i], tmp_9);
    i_0 = 3 * i + 2;
    tmp[i_0] = Base_Model_three_drones_P.Constant1_Value[i_0] - rtb_Force_Sum[2]
      * rtb_Force_Sum[i];
  }

  rtb_RL_dot_0 = 0.0;
  rtb_Gain = 0.0;
  rtb_RL_dot_1 = 0.0;
  Desired_Cable_Forces_Derivatives =
    Base_Model_three_drones_U.Desired_Cable_Forces_Derivatives[4];
  Desired_Cable_Forces_Derivatives_0 =
    Base_Model_three_drones_U.Desired_Cable_Forces_Derivatives[3];
  Desired_Cable_Forces_Derivatives_1 =
    Base_Model_three_drones_U.Desired_Cable_Forces_Derivatives[5];
  for (i = 0; i < 3; i++) {
    rtb_Product5[i] = ((tmp[i + 3] * Desired_Cable_Forces_Derivatives + tmp[i] *
                        Desired_Cable_Forces_Derivatives_0) + tmp[i + 6] *
                       Desired_Cable_Forces_Derivatives_1) / localProduct *
      Cable_Resting_Length;
    Desired_Cable_Forces = Attachment_Point_Vectors[i + 3];
    tmp_9 = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Transpose[3 * i]),
      _mm_set1_pd(Desired_Cable_Forces)), _mm_set_pd(rtb_Gain, rtb_RL_dot_0));
    _mm_storeu_pd(&tmp_7[0], tmp_9);
    rtb_RL_dot_0 = tmp_7[0];
    rtb_Gain = tmp_7[1];
    rtb_RL_dot_1 += rtb_Transpose[3 * i + 2] * Desired_Cable_Forces;
  }

  Desired_Cable_Forces = 0.0;
  for (i = 0; i < 3; i++) {
    rtb_pH2_dot[i] =
      ((Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[i + 3] *
        rtb_Gain + Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[i] *
        rtb_RL_dot_0) +
       Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[i + 6] *
       rtb_RL_dot_1) +
      Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[i];
    localProduct = Base_Model_three_drones_U.Desired_Cable_Forces[i + 6];
    Desired_Cable_Forces += localProduct * localProduct;
  }

  localProduct = sqrt(Desired_Cable_Forces);
  if (localProduct > Base_Model_three_drones_P.Saturation2_UpperSat) {
    localProduct = Base_Model_three_drones_P.Saturation2_UpperSat;
  } else if (localProduct < Base_Model_three_drones_P.Saturation2_LowerSat) {
    localProduct = Base_Model_three_drones_P.Saturation2_LowerSat;
  }

  _mm_storeu_pd(&rtb_Force_Sum[0], _mm_div_pd(_mm_loadu_pd
    (&Base_Model_three_drones_U.Desired_Cable_Forces[6]), _mm_set1_pd
    (localProduct)));
  rtb_Force_Sum[2] = Base_Model_three_drones_U.Desired_Cable_Forces[8] /
    localProduct;
  for (i = 0; i < 3; i++) {
    tmp_9 = _mm_sub_pd(_mm_loadu_pd(&Base_Model_three_drones_P.Constant2_Value[3
      * i]), _mm_mul_pd(_mm_loadu_pd(&rtb_Force_Sum[0]), _mm_set1_pd
                        (rtb_Force_Sum[i])));
    _mm_storeu_pd(&tmp[3 * i], tmp_9);
    i_0 = 3 * i + 2;
    tmp[i_0] = Base_Model_three_drones_P.Constant2_Value[i_0] - rtb_Force_Sum[2]
      * rtb_Force_Sum[i];
  }

  rtb_RL_dot_0 = 0.0;
  rtb_Gain = 0.0;
  rtb_RL_dot_1 = 0.0;
  Desired_Cable_Forces_Derivatives =
    Base_Model_three_drones_U.Desired_Cable_Forces_Derivatives[7];
  Desired_Cable_Forces_Derivatives_0 =
    Base_Model_three_drones_U.Desired_Cable_Forces_Derivatives[6];
  Desired_Cable_Forces_Derivatives_1 =
    Base_Model_three_drones_U.Desired_Cable_Forces_Derivatives[8];
  for (i = 0; i < 3; i++) {
    rtb_Product6[i] = ((tmp[i + 3] * Desired_Cable_Forces_Derivatives + tmp[i] *
                        Desired_Cable_Forces_Derivatives_0) + tmp[i + 6] *
                       Desired_Cable_Forces_Derivatives_1) / localProduct *
      Cable_Resting_Length;
    Desired_Cable_Forces = Attachment_Point_Vectors[i + 6];
    tmp_9 = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Transpose[3 * i]),
      _mm_set1_pd(Desired_Cable_Forces)), _mm_set_pd(rtb_Gain, rtb_RL_dot_0));
    _mm_storeu_pd(&tmp_7[0], tmp_9);
    rtb_RL_dot_0 = tmp_7[0];
    rtb_Gain = tmp_7[1];
    rtb_RL_dot_1 += rtb_Transpose[3 * i + 2] * Desired_Cable_Forces;
  }

  tmp[0] = Base_Model_three_drones_P.Constant_Value_m4;
  tmp[3] = Base_Model_three_drones_P.Gain_Gain_i * Attachment_Point_Vectors[2];
  tmp[6] = Attachment_Point_Vectors[1];
  tmp[1] = Attachment_Point_Vectors[2];
  tmp[4] = Base_Model_three_drones_P.Constant_Value_m4;
  _mm_storeu_pd(&tmp_7[0], _mm_mul_pd(_mm_set_pd
    (Base_Model_three_drones_P.Gain2_Gain_e,
     Base_Model_three_drones_P.Gain1_Gain_dl), _mm_loadu_pd
    (&Attachment_Point_Vectors[0])));
  tmp[7] = tmp_7[0];
  tmp[2] = tmp_7[1];
  tmp[5] = Attachment_Point_Vectors[0];
  tmp[8] = Base_Model_three_drones_P.Constant_Value_m4;
  tmp_1[0] = Base_Model_three_drones_P.Constant_Value_f;
  tmp_1[3] = Base_Model_three_drones_P.Gain_Gain_g * Attachment_Point_Vectors[5];
  tmp_1[6] = Attachment_Point_Vectors[4];
  tmp_1[1] = Attachment_Point_Vectors[5];
  tmp_1[4] = Base_Model_three_drones_P.Constant_Value_f;
  _mm_storeu_pd(&tmp_7[0], _mm_mul_pd(_mm_set_pd
    (Base_Model_three_drones_P.Gain2_Gain_a,
     Base_Model_three_drones_P.Gain1_Gain_g), _mm_loadu_pd
    (&Attachment_Point_Vectors[3])));
  tmp_1[7] = tmp_7[0];
  tmp_1[2] = tmp_7[1];
  tmp_1[5] = Attachment_Point_Vectors[3];
  tmp_1[8] = Base_Model_three_drones_P.Constant_Value_f;
  for (i = 0; i < 3; i++) {
    localProduct = Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[i];
    Desired_Cable_Forces = localProduct * rtb_RL_dot_0;
    rtb_Transpose[3 * i] = localProduct;
    localProduct = Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[i +
      3];
    Desired_Cable_Forces += localProduct * rtb_Gain;
    rtb_Transpose_tmp_0 = 3 * i + 1;
    rtb_Transpose[rtb_Transpose_tmp_0] = localProduct;
    localProduct = Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[i +
      6];
    rtb_Transpose_tmp = 3 * i + 2;
    rtb_Transpose[rtb_Transpose_tmp] = localProduct;
    rtb_Force_Sum[i] = (localProduct * rtb_RL_dot_1 + Desired_Cable_Forces) +
      Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[i];
    Desired_Cable_Forces = 0.0;
    localProduct = 0.0;
    Desired_Cable_Forces_Derivatives = 0.0;
    for (i_0 = 0; i_0 < 3; i_0++) {
      tmp_a = 3 * i + i_0;
      Desired_Cable_Forces_Derivatives_0 = rtb_Transpose[tmp_a];
      tmp_9 = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&tmp[3 * i_0]), _mm_set1_pd
        (Desired_Cable_Forces_Derivatives_0)), _mm_set_pd(localProduct,
        Desired_Cable_Forces));
      _mm_storeu_pd(&tmp_7[0], tmp_9);
      Desired_Cable_Forces = tmp_7[0];
      localProduct = tmp_7[1];
      Desired_Cable_Forces_Derivatives += tmp[3 * i_0 + 2] *
        Desired_Cable_Forces_Derivatives_0;
      tmp_2[tmp_a] = 0.0;
    }

    tmp_0[rtb_Transpose_tmp] = Desired_Cable_Forces_Derivatives;
    tmp_0[rtb_Transpose_tmp_0] = localProduct;
    tmp_0[3 * i] = Desired_Cable_Forces;
    Desired_Cable_Forces = tmp_2[3 * i];
    localProduct = tmp_2[rtb_Transpose_tmp_0];
    Desired_Cable_Forces_Derivatives = tmp_2[rtb_Transpose_tmp];
    for (i_0 = 0; i_0 < 3; i_0++) {
      Desired_Cable_Forces_Derivatives_0 = rtb_Transpose[3 * i + i_0];
      tmp_9 = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&tmp_1[3 * i_0]), _mm_set1_pd
        (Desired_Cable_Forces_Derivatives_0)), _mm_set_pd(localProduct,
        Desired_Cable_Forces));
      _mm_storeu_pd(&tmp_7[0], tmp_9);
      Desired_Cable_Forces = tmp_7[0];
      localProduct = tmp_7[1];
      Desired_Cable_Forces_Derivatives += tmp_1[3 * i_0 + 2] *
        Desired_Cable_Forces_Derivatives_0;
    }

    tmp_2[rtb_Transpose_tmp] = Desired_Cable_Forces_Derivatives;
    tmp_2[rtb_Transpose_tmp_0] = localProduct;
    tmp_2[3 * i] = Desired_Cable_Forces;
  }

  Desired_Cable_Forces = 0.0;
  localProduct = 0.0;
  Desired_Cable_Forces_Derivatives = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_9 = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&tmp_0[3 * i]), _mm_set1_pd
      (Base_Model_three_drones_U.Desired_Cable_Forces[i])), _mm_set_pd
                       (localProduct, Desired_Cable_Forces));
    _mm_storeu_pd(&tmp_7[0], tmp_9);
    Desired_Cable_Forces = tmp_7[0];
    localProduct = tmp_7[1];
    Desired_Cable_Forces_Derivatives += tmp_0[3 * i + 2] *
      Base_Model_three_drones_U.Desired_Cable_Forces[i];
    tmp_3[i] = 0.0;
  }

  rtb_Torque_Sum_0[2] = Desired_Cable_Forces_Derivatives;
  rtb_Torque_Sum_0[1] = localProduct;
  rtb_Torque_Sum_0[0] = Desired_Cable_Forces;
  tmp[0] = Base_Model_three_drones_P.Constant_Value_i;
  tmp[3] = Base_Model_three_drones_P.Gain_Gain_c * Attachment_Point_Vectors[8];
  tmp[6] = Attachment_Point_Vectors[7];
  tmp[1] = Attachment_Point_Vectors[8];
  tmp[4] = Base_Model_three_drones_P.Constant_Value_i;
  _mm_storeu_pd(&tmp_7[0], _mm_mul_pd(_mm_set_pd
    (Base_Model_three_drones_P.Gain2_Gain_m,
     Base_Model_three_drones_P.Gain1_Gain_f), _mm_loadu_pd
    (&Attachment_Point_Vectors[6])));
  tmp[7] = tmp_7[0];
  tmp[2] = tmp_7[1];
  tmp[5] = Attachment_Point_Vectors[6];
  tmp[8] = Base_Model_three_drones_P.Constant_Value_i;
  for (i = 0; i < 3; i++) {
    Desired_Cable_Forces = Base_Model_three_drones_U.Desired_Cable_Forces[i + 3];
    tmp_3[0] += tmp_2[3 * i] * Desired_Cable_Forces;
    i_0 = 3 * i + 1;
    tmp_3[1] += tmp_2[i_0] * Desired_Cable_Forces;
    tmp_a = 3 * i + 2;
    tmp_3[2] += tmp_2[tmp_a] * Desired_Cable_Forces;
    Desired_Cable_Forces = 0.0;
    localProduct = 0.0;
    Desired_Cable_Forces_Derivatives = 0.0;
    for (rtb_Transpose_tmp_0 = 0; rtb_Transpose_tmp_0 < 3; rtb_Transpose_tmp_0++)
    {
      Desired_Cable_Forces_Derivatives_0 = rtb_Transpose[3 * i +
        rtb_Transpose_tmp_0];
      tmp_9 = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&tmp[3 * rtb_Transpose_tmp_0]),
        _mm_set1_pd(Desired_Cable_Forces_Derivatives_0)), _mm_set_pd
                         (localProduct, Desired_Cable_Forces));
      _mm_storeu_pd(&tmp_7[0], tmp_9);
      Desired_Cable_Forces = tmp_7[0];
      localProduct = tmp_7[1];
      Desired_Cable_Forces_Derivatives += tmp[3 * rtb_Transpose_tmp_0 + 2] *
        Desired_Cable_Forces_Derivatives_0;
    }

    tmp_0[tmp_a] = Desired_Cable_Forces_Derivatives;
    tmp_0[i_0] = localProduct;
    tmp_0[3 * i] = Desired_Cable_Forces;
  }

  Desired_Cable_Forces = Base_Model_three_drones_U.Desired_Cable_Forces[7];
  rtb_RL_dot_0 = Base_Model_three_drones_U.Desired_Cable_Forces[6];
  rtb_Gain = Base_Model_three_drones_U.Desired_Cable_Forces[8];
  for (i = 0; i <= 0; i += 2) {
    tmp_9 = _mm_loadu_pd(&tmp_0[i + 3]);
    tmp_8 = _mm_loadu_pd(&tmp_0[i]);
    tmp_5 = _mm_loadu_pd(&tmp_0[i + 6]);
    tmp_6 = _mm_loadu_pd(&rtb_Torque_Sum_0[i]);
    tmp_4 = _mm_loadu_pd(&tmp_3[i]);
    _mm_storeu_pd(&rtb_Torque_Sum[i], _mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_9, _mm_set1_pd(Desired_Cable_Forces)), _mm_mul_pd(tmp_8,
      _mm_set1_pd(rtb_RL_dot_0))), _mm_mul_pd(tmp_5, _mm_set1_pd(rtb_Gain))),
      _mm_add_pd(tmp_6, tmp_4)));
    tmp_9 = _mm_loadu_pd
      (&Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[i]);
    _mm_storeu_pd(&rtb_Sum3[i], _mm_mul_pd(_mm_set1_pd
      (Base_Model_three_drones_P.Gain1_Gain_fl), tmp_9));
    _mm_storeu_pd(&rtb_MatrixMultiply5[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    rtb_Torque_Sum[i] = ((tmp_0[i + 3] * Desired_Cable_Forces + tmp_0[i] *
                          rtb_RL_dot_0) + tmp_0[i + 6] * rtb_Gain) +
      (rtb_Torque_Sum_0[i] + tmp_3[i]);
    rtb_Sum3[i] = Base_Model_three_drones_P.Gain1_Gain_fl *
      Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[i];
  }

  Desired_Cable_Forces = 0.0;
  rtb_RL_dot_0 = 0.0;
  rtb_RL_dot_1 = 0.0;
  for (i = 0; i < 3; i++) {
    _mm_storeu_pd(&tmp_7[0], _mm_add_pd(_mm_mul_pd(_mm_loadu_pd
      (&Load_Inertia_Matrix[3 * i]), _mm_set1_pd
      (Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[i])),
      _mm_set_pd(rtb_RL_dot_0, Desired_Cable_Forces)));
    Desired_Cable_Forces = tmp_7[0];
    rtb_RL_dot_0 = tmp_7[1];
    rtb_RL_dot_1 += Load_Inertia_Matrix[3 * i + 2] *
      Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[i];
  }

  rtb_Gain = -g * Load_Mass;
  utAssert(((((Load_Inertia_Matrix[0] * Load_Inertia_Matrix[4] *
               Load_Inertia_Matrix[8] - Load_Inertia_Matrix[0] *
               Load_Inertia_Matrix[5] * Load_Inertia_Matrix[7]) -
              Load_Inertia_Matrix[1] * Load_Inertia_Matrix[3] *
              Load_Inertia_Matrix[8]) + Load_Inertia_Matrix[2] *
             Load_Inertia_Matrix[3] * Load_Inertia_Matrix[7]) +
            Load_Inertia_Matrix[1] * Load_Inertia_Matrix[5] *
            Load_Inertia_Matrix[6]) - Load_Inertia_Matrix[2] *
           Load_Inertia_Matrix[4] * Load_Inertia_Matrix[6] != 0.0);
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE[0] +=
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_gainval *
    Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[0];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[0] +=
    ((((Base_Model_three_drones_U.Desired_Cable_Forces[0] +
        Base_Model_three_drones_U.Desired_Cable_Forces[3]) +
       Base_Model_three_drones_U.Desired_Cable_Forces[6]) + rtb_Gain *
      Base_Model_three_drones_P.Constant_Value_fy[0]) -
     Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[0] *
     Load_Damping_Coef) / Load_Mass *
    Base_Model_three_drones_P.DiscreteTimeIntegrator_gainval;
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE[1] +=
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_gainval *
    Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[1];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[1] +=
    ((((Base_Model_three_drones_U.Desired_Cable_Forces[1] +
        Base_Model_three_drones_U.Desired_Cable_Forces[4]) +
       Base_Model_three_drones_U.Desired_Cable_Forces[7]) + rtb_Gain *
      Base_Model_three_drones_P.Constant_Value_fy[1]) -
     Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[1] *
     Load_Damping_Coef) / Load_Mass *
    Base_Model_three_drones_P.DiscreteTimeIntegrator_gainval;
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE[2] +=
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_gainval *
    Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[2];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[2] +=
    ((((Base_Model_three_drones_U.Desired_Cable_Forces[2] +
        Base_Model_three_drones_U.Desired_Cable_Forces[5]) +
       Base_Model_three_drones_U.Desired_Cable_Forces[8]) + rtb_Gain *
      Base_Model_three_drones_P.Constant_Value_fy[2]) -
     Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[2] *
     Load_Damping_Coef) / Load_Mass *
    Base_Model_three_drones_P.DiscreteTimeIntegrator_gainval;
  for (i = 0; i <= 6; i += 2) {
    tmp_9 = _mm_loadu_pd(&rtb_RL_dot[i]);
    tmp_8 = _mm_loadu_pd
      (&Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[i]);
    _mm_storeu_pd(&Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[i],
                  _mm_add_pd(_mm_mul_pd(_mm_set1_pd
      (Base_Model_three_drones_P.DiscreteTimeIntegrator2_gainval), tmp_9), tmp_8));
  }

  for (i = 8; i < 9; i++) {
    Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[i] +=
      Base_Model_three_drones_P.DiscreteTimeIntegrator2_gainval * rtb_RL_dot[i];
  }

  rt_invd3x3_snf(Load_Inertia_Matrix, tmp);
  rtb_Torque_Sum_0[0] = ((rtb_Sum3[1] * rtb_RL_dot_1 - rtb_RL_dot_0 * rtb_Sum3[2])
    + rtb_Torque_Sum[0]) -
    Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[0] *
    Load_Damping_Coef;
  rtb_Gain = 0.0;
  rtb_Torque_Sum_0[1] = ((Desired_Cable_Forces * rtb_Sum3[2] - rtb_Sum3[0] *
    rtb_RL_dot_1) + rtb_Torque_Sum[1]) -
    Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[1] *
    Load_Damping_Coef;
  rtb_RL_dot_1 = 0.0;
  rtb_Torque_Sum_0[2] = ((rtb_Sum3[0] * rtb_RL_dot_0 - Desired_Cable_Forces *
    rtb_Sum3[1]) + rtb_Torque_Sum[2]) -
    Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[2] *
    Load_Damping_Coef;
  Desired_Cable_Forces = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_9 = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&tmp[3 * i]), _mm_set1_pd
      (rtb_Torque_Sum_0[i])), _mm_set_pd(rtb_RL_dot_1, rtb_Gain));
    _mm_storeu_pd(&tmp_7[0], tmp_9);
    rtb_Gain = tmp_7[0];
    rtb_RL_dot_1 = tmp_7[1];
    Desired_Cable_Forces += tmp[3 * i + 2] * rtb_Torque_Sum_0[i];
  }

  tmp_9 = _mm_set_pd(Base_Model_three_drones_P.DiscreteTimeIntegrator1_gainval_g,
                     Base_Model_three_drones_P.DiscreteTimeIntegrator3_gainval);
  _mm_storeu_pd(&tmp_7[0], _mm_add_pd(_mm_mul_pd(tmp_9, _mm_set_pd
    (Base_Model_three_drones_DW.UnitDelay_DSTATE[0], rtb_Gain)), _mm_set_pd
    (Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_f[0],
     Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[0])));
  Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[0] = tmp_7[0];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_f[0] = tmp_7[1];
  tmp_8 = _mm_set_pd(Base_Model_three_drones_P.DiscreteTimeIntegrator1_gainval_i,
                     Base_Model_three_drones_P.DiscreteTimeIntegrator1_gainval_n);
  _mm_storeu_pd(&tmp_7[0], _mm_add_pd(_mm_mul_pd(tmp_8, _mm_set_pd
    (Base_Model_three_drones_DW.UnitDelay_DSTATE_k[0],
     Base_Model_three_drones_DW.UnitDelay_DSTATE_b[0])), _mm_set_pd
    (Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_l[0],
     Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_g[0])));
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_g[0] = tmp_7[0];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_l[0] = tmp_7[1];
  _mm_storeu_pd(&tmp_7[0], _mm_add_pd(_mm_set_pd(rtb_Product5[0], rtb_Product4[0]),
    _mm_set_pd(rtb_pH2_dot[0], rtb_Transpose1[0])));
  Base_Model_three_drones_DW.UnitDelay_DSTATE[0] = tmp_7[0];
  Base_Model_three_drones_DW.UnitDelay_DSTATE_b[0] = tmp_7[1];
  Base_Model_three_drones_DW.UnitDelay_DSTATE_k[0] = rtb_Product6[0] +
    rtb_Force_Sum[0];
  _mm_storeu_pd(&tmp_7[0], _mm_add_pd(_mm_mul_pd(tmp_9, _mm_set_pd
    (Base_Model_three_drones_DW.UnitDelay_DSTATE[1], rtb_RL_dot_1)), _mm_set_pd
    (Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_f[1],
     Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[1])));
  Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[1] = tmp_7[0];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_f[1] = tmp_7[1];
  _mm_storeu_pd(&tmp_7[0], _mm_add_pd(_mm_mul_pd(tmp_8, _mm_set_pd
    (Base_Model_three_drones_DW.UnitDelay_DSTATE_k[1],
     Base_Model_three_drones_DW.UnitDelay_DSTATE_b[1])), _mm_set_pd
    (Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_l[1],
     Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_g[1])));
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_g[1] = tmp_7[0];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_l[1] = tmp_7[1];
  _mm_storeu_pd(&tmp_7[0], _mm_add_pd(_mm_set_pd(rtb_Product5[1], rtb_Product4[1]),
    _mm_set_pd(rtb_pH2_dot[1], rtb_Transpose1[1])));
  Base_Model_three_drones_DW.UnitDelay_DSTATE[1] = tmp_7[0];
  Base_Model_three_drones_DW.UnitDelay_DSTATE_b[1] = tmp_7[1];
  Base_Model_three_drones_DW.UnitDelay_DSTATE_k[1] = rtb_Product6[1] +
    rtb_Force_Sum[1];
  _mm_storeu_pd(&tmp_7[0], _mm_add_pd(_mm_mul_pd(tmp_9, _mm_set_pd
    (Base_Model_three_drones_DW.UnitDelay_DSTATE[2], Desired_Cable_Forces)),
    _mm_set_pd(Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_f[2],
               Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[2])));
  Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[2] = tmp_7[0];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_f[2] = tmp_7[1];
  _mm_storeu_pd(&tmp_7[0], _mm_add_pd(_mm_mul_pd(tmp_8, _mm_set_pd
    (Base_Model_three_drones_DW.UnitDelay_DSTATE_k[2],
     Base_Model_three_drones_DW.UnitDelay_DSTATE_b[2])), _mm_set_pd
    (Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_l[2],
     Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_g[2])));
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_g[2] = tmp_7[0];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_l[2] = tmp_7[1];
  _mm_storeu_pd(&tmp_7[0], _mm_add_pd(_mm_set_pd(rtb_Product5[2], rtb_Product4[2]),
    _mm_set_pd(rtb_pH2_dot[2], rtb_Transpose1[2])));
  Base_Model_three_drones_DW.UnitDelay_DSTATE[2] = tmp_7[0];
  Base_Model_three_drones_DW.UnitDelay_DSTATE_b[2] = tmp_7[1];
  Base_Model_three_drones_DW.UnitDelay_DSTATE_k[2] = rtb_Product6[2] +
    rtb_Force_Sum[2];
  fmu_LogOutput();
}

void Base_Model_three_drones_initialize(void)
{
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE[0] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_IC[0];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[0] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator_IC;
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE[1] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_IC[1];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[1] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator_IC;
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE[2] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_IC[2];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator_DSTATE[2] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator_IC;
  memcpy(&Base_Model_three_drones_DW.DiscreteTimeIntegrator2_DSTATE[0],
         &Base_Model_three_drones_P.DiscreteTimeIntegrator2_IC[0], 9U * sizeof
         (real_T));
  Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[0] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator3_IC;
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_f[0] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_IC_h[0];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_g[0] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_IC_e[0];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_l[0] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_IC_n[0];
  Base_Model_three_drones_DW.UnitDelay_DSTATE[0] =
    Base_Model_three_drones_P.UnitDelay_InitialCondition;
  Base_Model_three_drones_DW.UnitDelay_DSTATE_b[0] =
    Base_Model_three_drones_P.UnitDelay_InitialCondition_b;
  Base_Model_three_drones_DW.UnitDelay_DSTATE_k[0] =
    Base_Model_three_drones_P.UnitDelay_InitialCondition_o;
  Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[1] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator3_IC;
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_f[1] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_IC_h[1];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_g[1] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_IC_e[1];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_l[1] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_IC_n[1];
  Base_Model_three_drones_DW.UnitDelay_DSTATE[1] =
    Base_Model_three_drones_P.UnitDelay_InitialCondition;
  Base_Model_three_drones_DW.UnitDelay_DSTATE_b[1] =
    Base_Model_three_drones_P.UnitDelay_InitialCondition_b;
  Base_Model_three_drones_DW.UnitDelay_DSTATE_k[1] =
    Base_Model_three_drones_P.UnitDelay_InitialCondition_o;
  Base_Model_three_drones_DW.DiscreteTimeIntegrator3_DSTATE[2] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator3_IC;
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_f[2] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_IC_h[2];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_g[2] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_IC_e[2];
  Base_Model_three_drones_DW.DiscreteTimeIntegrator1_DSTATE_l[2] =
    Base_Model_three_drones_P.DiscreteTimeIntegrator1_IC_n[2];
  Base_Model_three_drones_DW.UnitDelay_DSTATE[2] =
    Base_Model_three_drones_P.UnitDelay_InitialCondition;
  Base_Model_three_drones_DW.UnitDelay_DSTATE_b[2] =
    Base_Model_three_drones_P.UnitDelay_InitialCondition_b;
  Base_Model_three_drones_DW.UnitDelay_DSTATE_k[2] =
    Base_Model_three_drones_P.UnitDelay_InitialCondition_o;
}

void Base_Model_three_drones_terminate(void)
{
}

RT_MODEL_Base_Model_three_drones_T *Base_Model_three_drones(void)
{
  (void) memset((void *)Base_Model_three_drones_M, 0,
                sizeof(RT_MODEL_Base_Model_three_drones_T));
  (void) memset((void *)&Base_Model_three_drones_DW, 0,
                sizeof(DW_Base_Model_three_drones_T));
  (void)memset(&Base_Model_three_drones_U, 0, sizeof
               (ExtU_Base_Model_three_drones_T));
  (void)memset(&Base_Model_three_drones_Y, 0, sizeof
               (ExtY_Base_Model_three_drones_T));
  return Base_Model_three_drones_M;
}
