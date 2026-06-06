#include "Base_Model_macros.h"
#include "Base_Model.h"
#include <string.h>
#include <emmintrin.h>
#include <math.h>
#include "rt_assert.h"
#include "Base_Model_private.h"
#include "rtwtypes.h"

real_T Attachment_Point_Vectors[12] = { 1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0,
  -1.0, 0.0, 1.0, -1.0, 0.0 } ;

real_T Cable_Resting_Length = 0.8;
real_T Kdc = 0.001;
real_T Kpc = 10.0;
real_T Load_Damping_Coef = 1.0;
real_T Load_Inertia_Matrix[9] = { 0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01
} ;

real_T Load_Mass = 0.7;
real_T g = 9.81;
DW_Base_Model_T Base_Model_DW;
ExtU_Base_Model_T Base_Model_U;
ExtY_Base_Model_T Base_Model_Y;
static RT_MODEL_Base_Model_T Base_Model_M_;
RT_MODEL_Base_Model_T *const Base_Model_M = &Base_Model_M_;
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

void Base_Model_step(void)
{
  __m128d tmp_4;
  __m128d tmp_5;
  __m128d tmp_6;
  __m128d tmp_7;
  __m128d tmp_8;
  __m128d tmp_9;
  __m128d tmp_b;
  __m128d tmp_c;
  real_T rtb_RL_dot[9];
  real_T rtb_Sum_f_tmp[9];
  real_T rtb_Sum_f_tmp_0[9];
  real_T rtb_Transpose[9];
  real_T tmp[9];
  real_T tmp_1[9];
  real_T tmp_2[9];
  real_T rtb_Force_Sum[3];
  real_T rtb_Force_Sum_0[3];
  real_T rtb_MatrixMultiply7[3];
  real_T rtb_MatrixMultiply7_0[3];
  real_T rtb_Sum_f_tmp_1[3];
  real_T rtb_Sum_p[3];
  real_T rtb_Transpose1[3];
  real_T tmp_0[3];
  real_T tmp_3[3];
  real_T tmp_a[2];
  real_T Attachment_Point_Vectors_0;
  real_T Desired_Cable_Forces_Derivatives;
  real_T Desired_Cable_Forces_Derivatives_0;
  real_T DiscreteTimeIntegrator2_DSTATE;
  real_T localProduct;
  real_T rtb_Gain;
  real_T rtb_MatrixMultiply5_idx_0;
  real_T rtb_MatrixMultiply5_idx_1;
  real_T rtb_MatrixMultiply5_idx_2;
  real_T rtb_RL_dot_0;
  real_T rtb_RL_dot_1;
  real_T rtb_RL_dot_2;
  real_T rtb_Sum_f_tmp_2;
  real_T rtb_Sum_f_tmp_3;
  real_T rtb_Sum_f_tmp_4;
  real_T rtb_Sum_f_tmp_5;
  real_T rtb_Sum_f_tmp_6;
  real_T rtb_Sum_f_tmp_7;
  int32_T i;
  int32_T i_0;
  int32_T rtb_Transpose_tmp;
  int32_T rtb_Transpose_tmp_0;
  int32_T tmp_d;
  Base_Model_Y.Load_Position[0] = Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[0];
  Base_Model_Y.Load_LinVelocity[0] =
    Base_Model_DW.DiscreteTimeIntegrator_DSTATE[0];
  Base_Model_Y.Load_Position[1] = Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[1];
  Base_Model_Y.Load_LinVelocity[1] =
    Base_Model_DW.DiscreteTimeIntegrator_DSTATE[1];
  Base_Model_Y.Load_Position[2] = Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[2];
  Base_Model_Y.Load_LinVelocity[2] =
    Base_Model_DW.DiscreteTimeIntegrator_DSTATE[2];
  memcpy(&Base_Model_Y.Load_Orientation[0],
         &Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[0], 9U * sizeof(real_T));
  tmp[0] = Base_Model_P.Constant_Value;
  tmp[3] = Base_Model_P.Gain_Gain *
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2];
  tmp[6] = Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[1];
  tmp[1] = Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2];
  tmp[4] = Base_Model_P.Constant_Value;
  tmp_c = _mm_mul_pd(_mm_set_pd(Base_Model_P.Gain2_Gain, Base_Model_P.Gain1_Gain),
                     _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator3_DSTATE
    [0]));
  _mm_storeu_pd(&tmp_a[0], tmp_c);
  tmp[7] = tmp_a[0];
  tmp[2] = tmp_a[1];
  tmp[5] = Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0];
  tmp[8] = Base_Model_P.Constant_Value;
  for (i = 0; i < 3; i++) {
    rtb_RL_dot_0 = 0.0;
    rtb_RL_dot_1 = 0.0;
    rtb_RL_dot_2 = 0.0;
    for (i_0 = 0; i_0 < 3; i_0++) {
      localProduct = tmp[3 * i + i_0];
      tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd
        (&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[3 * i_0]), _mm_set1_pd
        (localProduct)), _mm_set_pd(rtb_RL_dot_1, rtb_RL_dot_0));
      _mm_storeu_pd(&tmp_a[0], tmp_c);
      rtb_RL_dot_0 = tmp_a[0];
      rtb_RL_dot_1 = tmp_a[1];
      rtb_RL_dot_2 += Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[3 * i_0 + 2] *
        localProduct;
    }

    rtb_RL_dot[3 * i + 2] = rtb_RL_dot_2;
    rtb_RL_dot[3 * i + 1] = rtb_RL_dot_1;
    rtb_RL_dot[3 * i] = rtb_RL_dot_0;
  }

  memcpy(&Base_Model_Y.Load_Orientation_Derivative[0], &rtb_RL_dot[0], 9U *
         sizeof(real_T));
  Base_Model_Y.Load_AngVelocity[0] =
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0];
  Base_Model_Y.Drone_Positions[0] =
    Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[0];
  Base_Model_Y.Drone_Positions[3] =
    Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[0];
  Base_Model_Y.Drone_Positions[6] =
    Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[0];
  Base_Model_Y.Drone_Positions[9] =
    Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[0];
  Base_Model_Y.Drones_LinVelocity[0] = Base_Model_DW.UnitDelay_DSTATE[0];
  Base_Model_Y.Drones_LinVelocity[3] = Base_Model_DW.UnitDelay_DSTATE_b[0];
  Base_Model_Y.Drones_LinVelocity[6] = Base_Model_DW.UnitDelay_DSTATE_k[0];
  Base_Model_Y.Drones_LinVelocity[9] = Base_Model_DW.UnitDelay_DSTATE_c[0];
  Base_Model_Y.Load_AngVelocity[1] =
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[1];
  Base_Model_Y.Drone_Positions[1] =
    Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[1];
  Base_Model_Y.Drone_Positions[4] =
    Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[1];
  Base_Model_Y.Drone_Positions[7] =
    Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[1];
  Base_Model_Y.Drone_Positions[10] =
    Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[1];
  Base_Model_Y.Drones_LinVelocity[1] = Base_Model_DW.UnitDelay_DSTATE[1];
  Base_Model_Y.Drones_LinVelocity[4] = Base_Model_DW.UnitDelay_DSTATE_b[1];
  Base_Model_Y.Drones_LinVelocity[7] = Base_Model_DW.UnitDelay_DSTATE_k[1];
  Base_Model_Y.Drones_LinVelocity[10] = Base_Model_DW.UnitDelay_DSTATE_c[1];
  Base_Model_Y.Load_AngVelocity[2] =
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2];
  Base_Model_Y.Drone_Positions[2] =
    Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[2];
  Base_Model_Y.Drone_Positions[5] =
    Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[2];
  Base_Model_Y.Drone_Positions[8] =
    Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[2];
  Base_Model_Y.Drone_Positions[11] =
    Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[2];
  Base_Model_Y.Drones_LinVelocity[2] = Base_Model_DW.UnitDelay_DSTATE[2];
  Base_Model_Y.Drones_LinVelocity[5] = Base_Model_DW.UnitDelay_DSTATE_b[2];
  Base_Model_Y.Drones_LinVelocity[8] = Base_Model_DW.UnitDelay_DSTATE_k[2];
  Base_Model_Y.Drones_LinVelocity[11] = Base_Model_DW.UnitDelay_DSTATE_c[2];
  localProduct = sqrt((Base_Model_U.Desired_Cable_Forces[0] *
                       Base_Model_U.Desired_Cable_Forces[0] +
                       Base_Model_U.Desired_Cable_Forces[1] *
                       Base_Model_U.Desired_Cable_Forces[1]) +
                      Base_Model_U.Desired_Cable_Forces[2] *
                      Base_Model_U.Desired_Cable_Forces[2]);
  if (localProduct > Base_Model_P.Saturation_UpperSat) {
    localProduct = Base_Model_P.Saturation_UpperSat;
  } else if (localProduct < Base_Model_P.Saturation_LowerSat) {
    localProduct = Base_Model_P.Saturation_LowerSat;
  }

  tmp_c = _mm_set1_pd(localProduct);
  _mm_storeu_pd(&rtb_Transpose1[0], _mm_div_pd(_mm_loadu_pd
    (&Base_Model_U.Desired_Cable_Forces[0]), tmp_c));
  rtb_Transpose1[2] = Base_Model_U.Desired_Cable_Forces[2] / localProduct;
  rtb_Transpose[0] = Base_Model_P.Constant_Value_g;
  rtb_Transpose[3] = Base_Model_P.Gain_Gain_n *
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2];
  rtb_Transpose[6] = Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[1];
  rtb_Transpose[1] = Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2];
  rtb_Transpose[4] = Base_Model_P.Constant_Value_g;
  tmp_b = _mm_mul_pd(_mm_set_pd(Base_Model_P.Gain2_Gain_o,
    Base_Model_P.Gain1_Gain_d), _mm_loadu_pd
                     (&Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0]));
  _mm_storeu_pd(&tmp_a[0], tmp_b);
  rtb_Transpose[7] = tmp_a[0];
  rtb_Transpose[2] = tmp_a[1];
  rtb_Transpose[5] = Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0];
  rtb_Transpose[8] = Base_Model_P.Constant_Value_g;
  for (i = 0; i < 3; i++) {
    tmp_b = _mm_sub_pd(_mm_loadu_pd(&Base_Model_P.Constant_Value_m[3 * i]),
                       _mm_mul_pd(_mm_loadu_pd(&rtb_Transpose1[0]), _mm_set1_pd
      (rtb_Transpose1[i])));
    _mm_storeu_pd(&tmp[3 * i], tmp_b);
    i_0 = 3 * i + 2;
    tmp[i_0] = Base_Model_P.Constant_Value_m[i_0] - rtb_Transpose1[2] *
      rtb_Transpose1[i];
  }

  Desired_Cable_Forces_Derivatives =
    Base_Model_U.Desired_Cable_Forces_Derivatives[1];
  Desired_Cable_Forces_Derivatives_0 =
    Base_Model_U.Desired_Cable_Forces_Derivatives[0];
  rtb_MatrixMultiply5_idx_0 = Base_Model_U.Desired_Cable_Forces_Derivatives[2];
  for (i = 0; i <= 0; i += 2) {
    tmp_b = _mm_loadu_pd(&tmp[i + 3]);
    tmp_8 = _mm_loadu_pd(&tmp[i]);
    tmp_9 = _mm_loadu_pd(&tmp[i + 6]);
    _mm_storeu_pd(&rtb_Sum_f_tmp_1[i], _mm_div_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_b, _mm_set1_pd(Desired_Cable_Forces_Derivatives)),
       _mm_mul_pd(tmp_8, _mm_set1_pd(Desired_Cable_Forces_Derivatives_0))),
      _mm_mul_pd(tmp_9, _mm_set1_pd(rtb_MatrixMultiply5_idx_0))), tmp_c));
    _mm_storeu_pd(&rtb_MatrixMultiply7[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    rtb_Sum_f_tmp_1[i] = ((tmp[i + 3] * Desired_Cable_Forces_Derivatives + tmp[i]
      * Desired_Cable_Forces_Derivatives_0) + tmp[i + 6] *
                          rtb_MatrixMultiply5_idx_0) / localProduct;
  }

  Desired_Cable_Forces_Derivatives_0 = 0.0;
  rtb_MatrixMultiply5_idx_0 = 0.0;
  rtb_MatrixMultiply5_idx_1 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Transpose[3 * i]),
      _mm_set1_pd(Attachment_Point_Vectors[i])), _mm_set_pd
                       (rtb_MatrixMultiply5_idx_0,
                        Desired_Cable_Forces_Derivatives_0));
    _mm_storeu_pd(&tmp_a[0], tmp_c);
    Desired_Cable_Forces_Derivatives_0 = tmp_a[0];
    rtb_MatrixMultiply5_idx_0 = tmp_a[1];
    rtb_MatrixMultiply5_idx_1 += rtb_Transpose[3 * i + 2] *
      Attachment_Point_Vectors[i];
  }

  for (i = 0; i <= 0; i += 2) {
    tmp_c = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 3]);
    tmp_b = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i]);
    tmp_8 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 6]);
    tmp_9 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]);
    tmp_7 = _mm_loadu_pd(&rtb_Sum_f_tmp_1[i]);
    _mm_storeu_pd(&rtb_Force_Sum[i], _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_c, _mm_set1_pd(rtb_MatrixMultiply5_idx_0)), _mm_mul_pd
       (tmp_b, _mm_set1_pd(Desired_Cable_Forces_Derivatives_0))), _mm_mul_pd
      (tmp_8, _mm_set1_pd(rtb_MatrixMultiply5_idx_1))), tmp_9), _mm_mul_pd
      (_mm_set1_pd(Cable_Resting_Length), tmp_7)));
  }

  for (i = 2; i < 3; i++) {
    rtb_Force_Sum[i] = (((Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 3] *
                          rtb_MatrixMultiply5_idx_0 +
                          Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i] *
                          Desired_Cable_Forces_Derivatives_0) +
                         Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 6] *
                         rtb_MatrixMultiply5_idx_1) +
                        Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]) +
      Cable_Resting_Length * rtb_Sum_f_tmp_1[i];
  }

  for (i = 0; i <= 6; i += 2) {
    tmp_c = _mm_loadu_pd(&Base_Model_ConstP.pooled1[i]);
    _mm_storeu_pd(&rtb_Sum_f_tmp[i], _mm_mul_pd(_mm_set1_pd(Kpc), tmp_c));
    _mm_storeu_pd(&rtb_Sum_f_tmp_0[i], _mm_mul_pd(_mm_set1_pd(Kdc), tmp_c));
  }

  for (i = 8; i < 9; i++) {
    rtb_RL_dot_0 = Base_Model_ConstP.pooled1[i];
    rtb_Sum_f_tmp[i] = Kpc * rtb_RL_dot_0;
    rtb_Sum_f_tmp_0[i] = Kdc * rtb_RL_dot_0;
  }

  rtb_MatrixMultiply5_idx_2 = Attachment_Point_Vectors[1];
  rtb_Gain = Attachment_Point_Vectors[0];
  Attachment_Point_Vectors_0 = Attachment_Point_Vectors[2];
  for (i = 0; i <= 0; i += 2) {
    tmp_c = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 3]);
    tmp_b = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i]);
    tmp_8 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 6]);
    tmp_9 = _mm_loadu_pd(&rtb_Transpose1[i]);
    tmp_7 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]);
    tmp_6 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[i]);
    _mm_storeu_pd(&rtb_Sum_f_tmp_1[i], _mm_sub_pd(_mm_add_pd(_mm_add_pd
      (_mm_add_pd(_mm_add_pd(_mm_mul_pd(tmp_c, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_2)), _mm_mul_pd(tmp_b, _mm_set1_pd(rtb_Gain))),
                  _mm_mul_pd(tmp_8, _mm_set1_pd(Attachment_Point_Vectors_0))),
       _mm_mul_pd(tmp_9, _mm_set1_pd(Cable_Resting_Length))), tmp_7), tmp_6));
    tmp_c = _mm_loadu_pd(&rtb_Force_Sum[i]);
    tmp_b = _mm_loadu_pd(&Base_Model_DW.UnitDelay_DSTATE[i]);
    _mm_storeu_pd(&rtb_Force_Sum_0[i], _mm_sub_pd(tmp_c, tmp_b));
  }

  for (i = 2; i < 3; i++) {
    rtb_Sum_f_tmp_1[i] = ((((Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 3]
      * rtb_MatrixMultiply5_idx_2 +
      Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i] * rtb_Gain) +
      Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 6] *
      Attachment_Point_Vectors_0) + rtb_Transpose1[i] * Cable_Resting_Length) +
                          Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]) -
      Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[i];
    rtb_Force_Sum_0[i] = rtb_Force_Sum[i] - Base_Model_DW.UnitDelay_DSTATE[i];
  }

  localProduct = rtb_Sum_f_tmp_1[1];
  Desired_Cable_Forces_Derivatives = rtb_Sum_f_tmp_1[0];
  Desired_Cable_Forces_Derivatives_0 = rtb_Sum_f_tmp_1[2];
  for (i = 0; i <= 0; i += 2) {
    tmp_c = _mm_loadu_pd(&rtb_Sum_f_tmp[i + 3]);
    tmp_b = _mm_loadu_pd(&rtb_Sum_f_tmp[i]);
    tmp_8 = _mm_loadu_pd(&rtb_Sum_f_tmp[i + 6]);
    tmp_9 = _mm_loadu_pd(&rtb_Force_Sum[i]);
    _mm_storeu_pd(&rtb_Transpose1[i], _mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_c, _mm_set1_pd(localProduct)), _mm_mul_pd(tmp_b,
      _mm_set1_pd(Desired_Cable_Forces_Derivatives))), _mm_mul_pd(tmp_8,
      _mm_set1_pd(Desired_Cable_Forces_Derivatives_0))), tmp_9));
    _mm_storeu_pd(&rtb_Sum_f_tmp_1[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    rtb_Transpose1[i] = ((rtb_Sum_f_tmp[i + 3] * localProduct + rtb_Sum_f_tmp[i]
                          * Desired_Cable_Forces_Derivatives) + rtb_Sum_f_tmp[i
                         + 6] * Desired_Cable_Forces_Derivatives_0) +
      rtb_Force_Sum[i];
  }

  rtb_RL_dot_0 = 0.0;
  rtb_RL_dot_1 = 0.0;
  rtb_RL_dot_2 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Sum_f_tmp_0[3 * i]),
      _mm_set1_pd(rtb_Force_Sum_0[i])), _mm_set_pd(rtb_RL_dot_1, rtb_RL_dot_0));
    _mm_storeu_pd(&tmp_a[0], tmp_c);
    rtb_RL_dot_0 = tmp_a[0];
    rtb_RL_dot_1 = tmp_a[1];
    rtb_RL_dot_2 += rtb_Sum_f_tmp_0[3 * i + 2] * rtb_Force_Sum_0[i];
  }

  localProduct = sqrt((Base_Model_U.Desired_Cable_Forces[3] *
                       Base_Model_U.Desired_Cable_Forces[3] +
                       Base_Model_U.Desired_Cable_Forces[4] *
                       Base_Model_U.Desired_Cable_Forces[4]) +
                      Base_Model_U.Desired_Cable_Forces[5] *
                      Base_Model_U.Desired_Cable_Forces[5]);
  if (localProduct > Base_Model_P.Saturation1_UpperSat) {
    localProduct = Base_Model_P.Saturation1_UpperSat;
  } else if (localProduct < Base_Model_P.Saturation1_LowerSat) {
    localProduct = Base_Model_P.Saturation1_LowerSat;
  }

  tmp_c = _mm_set1_pd(localProduct);
  _mm_storeu_pd(&rtb_Force_Sum[0], _mm_div_pd(_mm_loadu_pd
    (&Base_Model_U.Desired_Cable_Forces[3]), tmp_c));
  rtb_Force_Sum[2] = Base_Model_U.Desired_Cable_Forces[5] / localProduct;
  for (i = 0; i < 3; i++) {
    tmp_b = _mm_sub_pd(_mm_loadu_pd(&Base_Model_P.Constant1_Value[3 * i]),
                       _mm_mul_pd(_mm_loadu_pd(&rtb_Force_Sum[0]), _mm_set1_pd
      (rtb_Force_Sum[i])));
    _mm_storeu_pd(&tmp[3 * i], tmp_b);
    i_0 = 3 * i + 2;
    tmp[i_0] = Base_Model_P.Constant1_Value[i_0] - rtb_Force_Sum[2] *
      rtb_Force_Sum[i];
  }

  Desired_Cable_Forces_Derivatives =
    Base_Model_U.Desired_Cable_Forces_Derivatives[4];
  Desired_Cable_Forces_Derivatives_0 =
    Base_Model_U.Desired_Cable_Forces_Derivatives[3];
  rtb_MatrixMultiply5_idx_0 = Base_Model_U.Desired_Cable_Forces_Derivatives[5];
  for (i = 0; i <= 0; i += 2) {
    tmp_b = _mm_loadu_pd(&tmp[i + 3]);
    tmp_8 = _mm_loadu_pd(&tmp[i]);
    tmp_9 = _mm_loadu_pd(&tmp[i + 6]);
    _mm_storeu_pd(&rtb_Sum_f_tmp_1[i], _mm_div_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_b, _mm_set1_pd(Desired_Cable_Forces_Derivatives)),
       _mm_mul_pd(tmp_8, _mm_set1_pd(Desired_Cable_Forces_Derivatives_0))),
      _mm_mul_pd(tmp_9, _mm_set1_pd(rtb_MatrixMultiply5_idx_0))), tmp_c));
    _mm_storeu_pd(&rtb_MatrixMultiply7[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    rtb_Sum_f_tmp_1[i] = ((tmp[i + 3] * Desired_Cable_Forces_Derivatives + tmp[i]
      * Desired_Cable_Forces_Derivatives_0) + tmp[i + 6] *
                          rtb_MatrixMultiply5_idx_0) / localProduct;
  }

  Desired_Cable_Forces_Derivatives_0 = 0.0;
  rtb_MatrixMultiply5_idx_0 = 0.0;
  rtb_MatrixMultiply5_idx_1 = 0.0;
  for (i = 0; i < 3; i++) {
    localProduct = Attachment_Point_Vectors[i + 3];
    tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Transpose[3 * i]),
      _mm_set1_pd(localProduct)), _mm_set_pd(rtb_MatrixMultiply5_idx_0,
      Desired_Cable_Forces_Derivatives_0));
    _mm_storeu_pd(&tmp_a[0], tmp_c);
    Desired_Cable_Forces_Derivatives_0 = tmp_a[0];
    rtb_MatrixMultiply5_idx_0 = tmp_a[1];
    rtb_MatrixMultiply5_idx_1 += rtb_Transpose[3 * i + 2] * localProduct;
  }

  rtb_MatrixMultiply5_idx_2 = Attachment_Point_Vectors[3];
  rtb_Gain = Attachment_Point_Vectors[4];
  Attachment_Point_Vectors_0 = Attachment_Point_Vectors[5];
  for (i = 0; i <= 0; i += 2) {
    tmp_c = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i]);
    tmp_b = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 3]);
    tmp_8 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 6]);
    tmp_9 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]);
    tmp_7 = _mm_loadu_pd(&rtb_Sum_f_tmp_1[i]);
    tmp_6 = _mm_set1_pd(Cable_Resting_Length);
    tmp_9 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(tmp_8, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_1)), _mm_add_pd(_mm_mul_pd(tmp_b, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_0)), _mm_mul_pd(tmp_c, _mm_set1_pd
      (Desired_Cable_Forces_Derivatives_0)))), tmp_9), _mm_mul_pd(tmp_6, tmp_7));
    _mm_storeu_pd(&rtb_MatrixMultiply7[i], tmp_9);
    tmp_7 = _mm_loadu_pd(&rtb_Force_Sum[i]);
    tmp_4 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]);
    tmp_5 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[i]);
    _mm_storeu_pd(&tmp_0[i], _mm_sub_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_8, _mm_set1_pd(Attachment_Point_Vectors_0)), _mm_add_pd
       (_mm_mul_pd(tmp_b, _mm_set1_pd(rtb_Gain)), _mm_mul_pd(tmp_c, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_2)))), _mm_mul_pd(tmp_7, tmp_6)), tmp_4), tmp_5));
    tmp_c = _mm_loadu_pd(&Base_Model_DW.UnitDelay_DSTATE_b[i]);
    _mm_storeu_pd(&rtb_Sum_p[i], _mm_sub_pd(tmp_9, tmp_c));
  }

  for (i = 2; i < 3; i++) {
    DiscreteTimeIntegrator2_DSTATE =
      Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i];
    localProduct = DiscreteTimeIntegrator2_DSTATE *
      Desired_Cable_Forces_Derivatives_0;
    Desired_Cable_Forces_Derivatives = DiscreteTimeIntegrator2_DSTATE *
      rtb_MatrixMultiply5_idx_2;
    DiscreteTimeIntegrator2_DSTATE =
      Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 3];
    localProduct += DiscreteTimeIntegrator2_DSTATE * rtb_MatrixMultiply5_idx_0;
    Desired_Cable_Forces_Derivatives += DiscreteTimeIntegrator2_DSTATE *
      rtb_Gain;
    DiscreteTimeIntegrator2_DSTATE =
      Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 6];
    localProduct = ((DiscreteTimeIntegrator2_DSTATE * rtb_MatrixMultiply5_idx_1
                     + localProduct) +
                    Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]) +
      Cable_Resting_Length * rtb_Sum_f_tmp_1[i];
    rtb_MatrixMultiply7[i] = localProduct;
    tmp_0[i] = (((DiscreteTimeIntegrator2_DSTATE * Attachment_Point_Vectors_0 +
                  Desired_Cable_Forces_Derivatives) + rtb_Force_Sum[i] *
                 Cable_Resting_Length) +
                Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]) -
      Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[i];
    rtb_Sum_p[i] = localProduct - Base_Model_DW.UnitDelay_DSTATE_b[i];
  }

  localProduct = tmp_0[1];
  Desired_Cable_Forces_Derivatives = tmp_0[0];
  Desired_Cable_Forces_Derivatives_0 = tmp_0[2];
  for (i = 0; i <= 0; i += 2) {
    tmp_c = _mm_loadu_pd(&rtb_Sum_f_tmp[i + 3]);
    tmp_b = _mm_loadu_pd(&rtb_Sum_f_tmp[i]);
    tmp_8 = _mm_loadu_pd(&rtb_Sum_f_tmp[i + 6]);
    tmp_9 = _mm_loadu_pd(&rtb_MatrixMultiply7[i]);
    _mm_storeu_pd(&rtb_Force_Sum_0[i], _mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_c, _mm_set1_pd(localProduct)), _mm_mul_pd(tmp_b,
      _mm_set1_pd(Desired_Cable_Forces_Derivatives))), _mm_mul_pd(tmp_8,
      _mm_set1_pd(Desired_Cable_Forces_Derivatives_0))), tmp_9));
    _mm_storeu_pd(&rtb_Sum_f_tmp_1[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    rtb_Force_Sum_0[i] = ((rtb_Sum_f_tmp[i + 3] * localProduct + rtb_Sum_f_tmp[i]
      * Desired_Cable_Forces_Derivatives) + rtb_Sum_f_tmp[i + 6] *
                          Desired_Cable_Forces_Derivatives_0) +
      rtb_MatrixMultiply7[i];
  }

  rtb_Sum_f_tmp_5 = 0.0;
  rtb_Sum_f_tmp_6 = 0.0;
  rtb_Sum_f_tmp_7 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Sum_f_tmp_0[3 * i]),
      _mm_set1_pd(rtb_Sum_p[i])), _mm_set_pd(rtb_Sum_f_tmp_6, rtb_Sum_f_tmp_5));
    _mm_storeu_pd(&tmp_a[0], tmp_c);
    rtb_Sum_f_tmp_5 = tmp_a[0];
    rtb_Sum_f_tmp_6 = tmp_a[1];
    rtb_Sum_f_tmp_7 += rtb_Sum_f_tmp_0[3 * i + 2] * rtb_Sum_p[i];
  }

  localProduct = sqrt((Base_Model_U.Desired_Cable_Forces[6] *
                       Base_Model_U.Desired_Cable_Forces[6] +
                       Base_Model_U.Desired_Cable_Forces[7] *
                       Base_Model_U.Desired_Cable_Forces[7]) +
                      Base_Model_U.Desired_Cable_Forces[8] *
                      Base_Model_U.Desired_Cable_Forces[8]);
  if (localProduct > Base_Model_P.Saturation2_UpperSat) {
    localProduct = Base_Model_P.Saturation2_UpperSat;
  } else if (localProduct < Base_Model_P.Saturation2_LowerSat) {
    localProduct = Base_Model_P.Saturation2_LowerSat;
  }

  tmp_c = _mm_set1_pd(localProduct);
  _mm_storeu_pd(&rtb_Force_Sum[0], _mm_div_pd(_mm_loadu_pd
    (&Base_Model_U.Desired_Cable_Forces[6]), tmp_c));
  rtb_Force_Sum[2] = Base_Model_U.Desired_Cable_Forces[8] / localProduct;
  for (i = 0; i < 3; i++) {
    tmp_b = _mm_sub_pd(_mm_loadu_pd(&Base_Model_P.Constant2_Value[3 * i]),
                       _mm_mul_pd(_mm_loadu_pd(&rtb_Force_Sum[0]), _mm_set1_pd
      (rtb_Force_Sum[i])));
    _mm_storeu_pd(&tmp[3 * i], tmp_b);
    i_0 = 3 * i + 2;
    tmp[i_0] = Base_Model_P.Constant2_Value[i_0] - rtb_Force_Sum[2] *
      rtb_Force_Sum[i];
  }

  Desired_Cable_Forces_Derivatives =
    Base_Model_U.Desired_Cable_Forces_Derivatives[7];
  Desired_Cable_Forces_Derivatives_0 =
    Base_Model_U.Desired_Cable_Forces_Derivatives[6];
  rtb_MatrixMultiply5_idx_0 = Base_Model_U.Desired_Cable_Forces_Derivatives[8];
  for (i = 0; i <= 0; i += 2) {
    tmp_b = _mm_loadu_pd(&tmp[i + 3]);
    tmp_8 = _mm_loadu_pd(&tmp[i]);
    tmp_9 = _mm_loadu_pd(&tmp[i + 6]);
    _mm_storeu_pd(&rtb_Sum_f_tmp_1[i], _mm_div_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_b, _mm_set1_pd(Desired_Cable_Forces_Derivatives)),
       _mm_mul_pd(tmp_8, _mm_set1_pd(Desired_Cable_Forces_Derivatives_0))),
      _mm_mul_pd(tmp_9, _mm_set1_pd(rtb_MatrixMultiply5_idx_0))), tmp_c));
    _mm_storeu_pd(&rtb_MatrixMultiply7[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    rtb_Sum_f_tmp_1[i] = ((tmp[i + 3] * Desired_Cable_Forces_Derivatives + tmp[i]
      * Desired_Cable_Forces_Derivatives_0) + tmp[i + 6] *
                          rtb_MatrixMultiply5_idx_0) / localProduct;
  }

  Desired_Cable_Forces_Derivatives_0 = 0.0;
  rtb_MatrixMultiply5_idx_0 = 0.0;
  rtb_MatrixMultiply5_idx_1 = 0.0;
  for (i = 0; i < 3; i++) {
    localProduct = Attachment_Point_Vectors[i + 6];
    tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Transpose[3 * i]),
      _mm_set1_pd(localProduct)), _mm_set_pd(rtb_MatrixMultiply5_idx_0,
      Desired_Cable_Forces_Derivatives_0));
    _mm_storeu_pd(&tmp_a[0], tmp_c);
    Desired_Cable_Forces_Derivatives_0 = tmp_a[0];
    rtb_MatrixMultiply5_idx_0 = tmp_a[1];
    rtb_MatrixMultiply5_idx_1 += rtb_Transpose[3 * i + 2] * localProduct;
  }

  rtb_MatrixMultiply5_idx_2 = Attachment_Point_Vectors[6];
  rtb_Gain = Attachment_Point_Vectors[7];
  Attachment_Point_Vectors_0 = Attachment_Point_Vectors[8];
  for (i = 0; i <= 0; i += 2) {
    tmp_c = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i]);
    tmp_b = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 3]);
    tmp_8 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 6]);
    tmp_9 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]);
    tmp_7 = _mm_loadu_pd(&rtb_Sum_f_tmp_1[i]);
    tmp_6 = _mm_set1_pd(Cable_Resting_Length);
    tmp_9 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(tmp_8, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_1)), _mm_add_pd(_mm_mul_pd(tmp_b, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_0)), _mm_mul_pd(tmp_c, _mm_set1_pd
      (Desired_Cable_Forces_Derivatives_0)))), tmp_9), _mm_mul_pd(tmp_6, tmp_7));
    _mm_storeu_pd(&rtb_MatrixMultiply7[i], tmp_9);
    tmp_7 = _mm_loadu_pd(&rtb_Force_Sum[i]);
    tmp_4 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]);
    tmp_5 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[i]);
    _mm_storeu_pd(&tmp_0[i], _mm_sub_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_8, _mm_set1_pd(Attachment_Point_Vectors_0)), _mm_add_pd
       (_mm_mul_pd(tmp_b, _mm_set1_pd(rtb_Gain)), _mm_mul_pd(tmp_c, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_2)))), _mm_mul_pd(tmp_7, tmp_6)), tmp_4), tmp_5));
    tmp_c = _mm_loadu_pd(&Base_Model_DW.UnitDelay_DSTATE_k[i]);
    _mm_storeu_pd(&rtb_Sum_p[i], _mm_sub_pd(tmp_9, tmp_c));
  }

  for (i = 2; i < 3; i++) {
    DiscreteTimeIntegrator2_DSTATE =
      Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i];
    localProduct = DiscreteTimeIntegrator2_DSTATE *
      Desired_Cable_Forces_Derivatives_0;
    Desired_Cable_Forces_Derivatives = DiscreteTimeIntegrator2_DSTATE *
      rtb_MatrixMultiply5_idx_2;
    DiscreteTimeIntegrator2_DSTATE =
      Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 3];
    localProduct += DiscreteTimeIntegrator2_DSTATE * rtb_MatrixMultiply5_idx_0;
    Desired_Cable_Forces_Derivatives += DiscreteTimeIntegrator2_DSTATE *
      rtb_Gain;
    DiscreteTimeIntegrator2_DSTATE =
      Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 6];
    localProduct = ((DiscreteTimeIntegrator2_DSTATE * rtb_MatrixMultiply5_idx_1
                     + localProduct) +
                    Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]) +
      Cable_Resting_Length * rtb_Sum_f_tmp_1[i];
    rtb_MatrixMultiply7[i] = localProduct;
    tmp_0[i] = (((DiscreteTimeIntegrator2_DSTATE * Attachment_Point_Vectors_0 +
                  Desired_Cable_Forces_Derivatives) + rtb_Force_Sum[i] *
                 Cable_Resting_Length) +
                Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]) -
      Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[i];
    rtb_Sum_p[i] = localProduct - Base_Model_DW.UnitDelay_DSTATE_k[i];
  }

  localProduct = tmp_0[1];
  Desired_Cable_Forces_Derivatives = tmp_0[0];
  Desired_Cable_Forces_Derivatives_0 = tmp_0[2];
  for (i = 0; i <= 0; i += 2) {
    tmp_c = _mm_loadu_pd(&rtb_Sum_f_tmp[i + 3]);
    tmp_b = _mm_loadu_pd(&rtb_Sum_f_tmp[i]);
    tmp_8 = _mm_loadu_pd(&rtb_Sum_f_tmp[i + 6]);
    tmp_9 = _mm_loadu_pd(&rtb_MatrixMultiply7[i]);
    _mm_storeu_pd(&rtb_MatrixMultiply7_0[i], _mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_c, _mm_set1_pd(localProduct)), _mm_mul_pd(tmp_b,
      _mm_set1_pd(Desired_Cable_Forces_Derivatives))), _mm_mul_pd(tmp_8,
      _mm_set1_pd(Desired_Cable_Forces_Derivatives_0))), tmp_9));
    _mm_storeu_pd(&rtb_Sum_f_tmp_1[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    rtb_MatrixMultiply7_0[i] = ((rtb_Sum_f_tmp[i + 3] * localProduct +
      rtb_Sum_f_tmp[i] * Desired_Cable_Forces_Derivatives) + rtb_Sum_f_tmp[i + 6]
      * Desired_Cable_Forces_Derivatives_0) + rtb_MatrixMultiply7[i];
  }

  rtb_Sum_f_tmp_2 = 0.0;
  rtb_Sum_f_tmp_3 = 0.0;
  rtb_Sum_f_tmp_4 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Sum_f_tmp_0[3 * i]),
      _mm_set1_pd(rtb_Sum_p[i])), _mm_set_pd(rtb_Sum_f_tmp_3, rtb_Sum_f_tmp_2));
    _mm_storeu_pd(&tmp_a[0], tmp_c);
    rtb_Sum_f_tmp_2 = tmp_a[0];
    rtb_Sum_f_tmp_3 = tmp_a[1];
    rtb_Sum_f_tmp_4 += rtb_Sum_f_tmp_0[3 * i + 2] * rtb_Sum_p[i];
  }

  localProduct = sqrt((Base_Model_U.Desired_Cable_Forces[9] *
                       Base_Model_U.Desired_Cable_Forces[9] +
                       Base_Model_U.Desired_Cable_Forces[10] *
                       Base_Model_U.Desired_Cable_Forces[10]) +
                      Base_Model_U.Desired_Cable_Forces[11] *
                      Base_Model_U.Desired_Cable_Forces[11]);
  if (localProduct > Base_Model_P.Saturation3_UpperSat) {
    localProduct = Base_Model_P.Saturation3_UpperSat;
  } else if (localProduct < Base_Model_P.Saturation3_LowerSat) {
    localProduct = Base_Model_P.Saturation3_LowerSat;
  }

  tmp_c = _mm_set1_pd(localProduct);
  _mm_storeu_pd(&rtb_Force_Sum[0], _mm_div_pd(_mm_loadu_pd
    (&Base_Model_U.Desired_Cable_Forces[9]), tmp_c));
  rtb_Force_Sum[2] = Base_Model_U.Desired_Cable_Forces[11] / localProduct;
  for (i = 0; i < 3; i++) {
    tmp_b = _mm_sub_pd(_mm_loadu_pd(&Base_Model_P.Constant3_Value[3 * i]),
                       _mm_mul_pd(_mm_loadu_pd(&rtb_Force_Sum[0]), _mm_set1_pd
      (rtb_Force_Sum[i])));
    _mm_storeu_pd(&tmp[3 * i], tmp_b);
    i_0 = 3 * i + 2;
    tmp[i_0] = Base_Model_P.Constant3_Value[i_0] - rtb_Force_Sum[2] *
      rtb_Force_Sum[i];
  }

  Desired_Cable_Forces_Derivatives =
    Base_Model_U.Desired_Cable_Forces_Derivatives[10];
  Desired_Cable_Forces_Derivatives_0 =
    Base_Model_U.Desired_Cable_Forces_Derivatives[9];
  rtb_MatrixMultiply5_idx_0 = Base_Model_U.Desired_Cable_Forces_Derivatives[11];
  for (i = 0; i <= 0; i += 2) {
    tmp_b = _mm_loadu_pd(&tmp[i + 3]);
    tmp_8 = _mm_loadu_pd(&tmp[i]);
    tmp_9 = _mm_loadu_pd(&tmp[i + 6]);
    _mm_storeu_pd(&rtb_Sum_f_tmp_1[i], _mm_div_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_b, _mm_set1_pd(Desired_Cable_Forces_Derivatives)),
       _mm_mul_pd(tmp_8, _mm_set1_pd(Desired_Cable_Forces_Derivatives_0))),
      _mm_mul_pd(tmp_9, _mm_set1_pd(rtb_MatrixMultiply5_idx_0))), tmp_c));
    _mm_storeu_pd(&rtb_MatrixMultiply7[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    rtb_Sum_f_tmp_1[i] = ((tmp[i + 3] * Desired_Cable_Forces_Derivatives + tmp[i]
      * Desired_Cable_Forces_Derivatives_0) + tmp[i + 6] *
                          rtb_MatrixMultiply5_idx_0) / localProduct;
  }

  Desired_Cable_Forces_Derivatives_0 = 0.0;
  rtb_MatrixMultiply5_idx_0 = 0.0;
  rtb_MatrixMultiply5_idx_1 = 0.0;
  for (i = 0; i < 3; i++) {
    localProduct = Attachment_Point_Vectors[i + 9];
    tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Transpose[3 * i]),
      _mm_set1_pd(localProduct)), _mm_set_pd(rtb_MatrixMultiply5_idx_0,
      Desired_Cable_Forces_Derivatives_0));
    _mm_storeu_pd(&tmp_a[0], tmp_c);
    Desired_Cable_Forces_Derivatives_0 = tmp_a[0];
    rtb_MatrixMultiply5_idx_0 = tmp_a[1];
    rtb_MatrixMultiply5_idx_1 += rtb_Transpose[3 * i + 2] * localProduct;
  }

  rtb_MatrixMultiply5_idx_2 = Attachment_Point_Vectors[9];
  rtb_Gain = Attachment_Point_Vectors[10];
  Attachment_Point_Vectors_0 = Attachment_Point_Vectors[11];
  for (i = 0; i <= 0; i += 2) {
    tmp_c = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i]);
    tmp_b = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 3]);
    tmp_8 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 6]);
    tmp_9 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]);
    tmp_7 = _mm_loadu_pd(&rtb_Sum_f_tmp_1[i]);
    tmp_6 = _mm_set1_pd(Cable_Resting_Length);
    tmp_9 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(tmp_8, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_1)), _mm_add_pd(_mm_mul_pd(tmp_b, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_0)), _mm_mul_pd(tmp_c, _mm_set1_pd
      (Desired_Cable_Forces_Derivatives_0)))), tmp_9), _mm_mul_pd(tmp_6, tmp_7));
    _mm_storeu_pd(&rtb_MatrixMultiply7[i], tmp_9);
    tmp_7 = _mm_loadu_pd(&rtb_Force_Sum[i]);
    tmp_4 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]);
    tmp_5 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[i]);
    _mm_storeu_pd(&tmp_0[i], _mm_sub_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_8, _mm_set1_pd(Attachment_Point_Vectors_0)), _mm_add_pd
       (_mm_mul_pd(tmp_b, _mm_set1_pd(rtb_Gain)), _mm_mul_pd(tmp_c, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_2)))), _mm_mul_pd(tmp_7, tmp_6)), tmp_4), tmp_5));
    tmp_c = _mm_loadu_pd(&Base_Model_DW.UnitDelay_DSTATE_c[i]);
    _mm_storeu_pd(&rtb_Sum_p[i], _mm_sub_pd(tmp_9, tmp_c));
  }

  for (i = 2; i < 3; i++) {
    DiscreteTimeIntegrator2_DSTATE =
      Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i];
    localProduct = DiscreteTimeIntegrator2_DSTATE *
      Desired_Cable_Forces_Derivatives_0;
    Desired_Cable_Forces_Derivatives = DiscreteTimeIntegrator2_DSTATE *
      rtb_MatrixMultiply5_idx_2;
    DiscreteTimeIntegrator2_DSTATE =
      Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 3];
    localProduct += DiscreteTimeIntegrator2_DSTATE * rtb_MatrixMultiply5_idx_0;
    Desired_Cable_Forces_Derivatives += DiscreteTimeIntegrator2_DSTATE *
      rtb_Gain;
    DiscreteTimeIntegrator2_DSTATE =
      Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 6];
    localProduct = ((DiscreteTimeIntegrator2_DSTATE * rtb_MatrixMultiply5_idx_1
                     + localProduct) +
                    Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]) +
      Cable_Resting_Length * rtb_Sum_f_tmp_1[i];
    rtb_MatrixMultiply7[i] = localProduct;
    tmp_0[i] = (((DiscreteTimeIntegrator2_DSTATE * Attachment_Point_Vectors_0 +
                  Desired_Cable_Forces_Derivatives) + rtb_Force_Sum[i] *
                 Cable_Resting_Length) +
                Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]) -
      Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[i];
    rtb_Sum_p[i] = localProduct - Base_Model_DW.UnitDelay_DSTATE_c[i];
  }

  localProduct = tmp_0[1];
  Desired_Cable_Forces_Derivatives = tmp_0[0];
  Desired_Cable_Forces_Derivatives_0 = tmp_0[2];
  for (i = 0; i <= 0; i += 2) {
    tmp_c = _mm_loadu_pd(&rtb_Sum_f_tmp[i + 3]);
    tmp_b = _mm_loadu_pd(&rtb_Sum_f_tmp[i]);
    tmp_8 = _mm_loadu_pd(&rtb_Sum_f_tmp[i + 6]);
    tmp_9 = _mm_loadu_pd(&rtb_MatrixMultiply7[i]);
    _mm_storeu_pd(&rtb_Force_Sum[i], _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd
      (tmp_c, _mm_set1_pd(localProduct)), _mm_mul_pd(tmp_b, _mm_set1_pd
      (Desired_Cable_Forces_Derivatives))), _mm_mul_pd(tmp_8, _mm_set1_pd
      (Desired_Cable_Forces_Derivatives_0))), tmp_9));
    _mm_storeu_pd(&rtb_Sum_f_tmp_1[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    rtb_Force_Sum[i] = ((rtb_Sum_f_tmp[i + 3] * localProduct + rtb_Sum_f_tmp[i] *
                         Desired_Cable_Forces_Derivatives) + rtb_Sum_f_tmp[i + 6]
                        * Desired_Cable_Forces_Derivatives_0) +
      rtb_MatrixMultiply7[i];
  }

  localProduct = 0.0;
  Desired_Cable_Forces_Derivatives = 0.0;
  Desired_Cable_Forces_Derivatives_0 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Sum_f_tmp_0[3 * i]),
      _mm_set1_pd(rtb_Sum_p[i])), _mm_set_pd(Desired_Cable_Forces_Derivatives,
      localProduct));
    _mm_storeu_pd(&tmp_a[0], tmp_c);
    localProduct = tmp_a[0];
    Desired_Cable_Forces_Derivatives = tmp_a[1];
    Desired_Cable_Forces_Derivatives_0 += rtb_Sum_f_tmp_0[3 * i + 2] *
      rtb_Sum_p[i];
  }

  rtb_Sum_f_tmp_1[2] = Desired_Cable_Forces_Derivatives_0;
  rtb_Sum_f_tmp_1[1] = Desired_Cable_Forces_Derivatives;
  rtb_Sum_f_tmp_1[0] = localProduct;
  tmp[0] = Base_Model_P.Constant_Value_m4;
  tmp[3] = Base_Model_P.Gain_Gain_i * Attachment_Point_Vectors[2];
  tmp[6] = Attachment_Point_Vectors[1];
  tmp[1] = Attachment_Point_Vectors[2];
  tmp[4] = Base_Model_P.Constant_Value_m4;
  _mm_storeu_pd(&tmp_a[0], _mm_mul_pd(_mm_set_pd(Base_Model_P.Gain2_Gain_e,
    Base_Model_P.Gain1_Gain_dl), _mm_loadu_pd(&Attachment_Point_Vectors[0])));
  tmp[7] = tmp_a[0];
  tmp[2] = tmp_a[1];
  tmp[5] = Attachment_Point_Vectors[0];
  tmp[8] = Base_Model_P.Constant_Value_m4;
  rtb_Sum_f_tmp_0[0] = Base_Model_P.Constant_Value_f;
  rtb_Sum_f_tmp_0[3] = Base_Model_P.Gain_Gain_g * Attachment_Point_Vectors[5];
  rtb_Sum_f_tmp_0[6] = Attachment_Point_Vectors[4];
  rtb_Sum_f_tmp_0[1] = Attachment_Point_Vectors[5];
  rtb_Sum_f_tmp_0[4] = Base_Model_P.Constant_Value_f;
  _mm_storeu_pd(&tmp_a[0], _mm_mul_pd(_mm_set_pd(Base_Model_P.Gain2_Gain_a,
    Base_Model_P.Gain1_Gain_g), _mm_loadu_pd(&Attachment_Point_Vectors[3])));
  rtb_Sum_f_tmp_0[7] = tmp_a[0];
  rtb_Sum_f_tmp_0[2] = tmp_a[1];
  rtb_Sum_f_tmp_0[5] = Attachment_Point_Vectors[3];
  rtb_Sum_f_tmp_0[8] = Base_Model_P.Constant_Value_f;
  for (i = 0; i < 3; i++) {
    rtb_Sum_p[i] = rtb_Force_Sum[i] + rtb_Sum_f_tmp_1[i];
    rtb_Transpose[3 * i] = Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i];
    rtb_Transpose_tmp_0 = 3 * i + 1;
    rtb_Transpose[rtb_Transpose_tmp_0] =
      Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 3];
    rtb_Transpose_tmp = 3 * i + 2;
    rtb_Transpose[rtb_Transpose_tmp] =
      Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i + 6];
    localProduct = 0.0;
    Desired_Cable_Forces_Derivatives = 0.0;
    Desired_Cable_Forces_Derivatives_0 = 0.0;
    for (i_0 = 0; i_0 < 3; i_0++) {
      tmp_d = 3 * i + i_0;
      rtb_Gain = rtb_Transpose[tmp_d];
      tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&tmp[3 * i_0]), _mm_set1_pd
        (rtb_Gain)), _mm_set_pd(Desired_Cable_Forces_Derivatives, localProduct));
      _mm_storeu_pd(&tmp_a[0], tmp_c);
      localProduct = tmp_a[0];
      Desired_Cable_Forces_Derivatives = tmp_a[1];
      Desired_Cable_Forces_Derivatives_0 += tmp[3 * i_0 + 2] * rtb_Gain;
      tmp_1[tmp_d] = 0.0;
    }

    rtb_Sum_f_tmp[rtb_Transpose_tmp] = Desired_Cable_Forces_Derivatives_0;
    rtb_Sum_f_tmp[rtb_Transpose_tmp_0] = Desired_Cable_Forces_Derivatives;
    rtb_Sum_f_tmp[3 * i] = localProduct;
    localProduct = tmp_1[3 * i];
    Desired_Cable_Forces_Derivatives = tmp_1[rtb_Transpose_tmp_0];
    Desired_Cable_Forces_Derivatives_0 = tmp_1[rtb_Transpose_tmp];
    for (i_0 = 0; i_0 < 3; i_0++) {
      rtb_Gain = rtb_Transpose[3 * i + i_0];
      tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Sum_f_tmp_0[3 * i_0]),
        _mm_set1_pd(rtb_Gain)), _mm_set_pd(Desired_Cable_Forces_Derivatives,
        localProduct));
      _mm_storeu_pd(&tmp_a[0], tmp_c);
      localProduct = tmp_a[0];
      Desired_Cable_Forces_Derivatives = tmp_a[1];
      Desired_Cable_Forces_Derivatives_0 += rtb_Sum_f_tmp_0[3 * i_0 + 2] *
        rtb_Gain;
    }

    tmp_1[rtb_Transpose_tmp] = Desired_Cable_Forces_Derivatives_0;
    tmp_1[rtb_Transpose_tmp_0] = Desired_Cable_Forces_Derivatives;
    tmp_1[3 * i] = localProduct;
  }

  localProduct = 0.0;
  Desired_Cable_Forces_Derivatives = 0.0;
  Desired_Cable_Forces_Derivatives_0 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Sum_f_tmp[3 * i]),
      _mm_set1_pd(Base_Model_U.Desired_Cable_Forces[i])), _mm_set_pd
                       (Desired_Cable_Forces_Derivatives, localProduct));
    _mm_storeu_pd(&tmp_a[0], tmp_c);
    localProduct = tmp_a[0];
    Desired_Cable_Forces_Derivatives = tmp_a[1];
    Desired_Cable_Forces_Derivatives_0 += rtb_Sum_f_tmp[3 * i + 2] *
      Base_Model_U.Desired_Cable_Forces[i];
    tmp_0[i] = 0.0;
  }

  rtb_Sum_f_tmp_1[2] = Desired_Cable_Forces_Derivatives_0;
  rtb_Sum_f_tmp_1[1] = Desired_Cable_Forces_Derivatives;
  rtb_Sum_f_tmp_1[0] = localProduct;
  tmp[0] = Base_Model_P.Constant_Value_i;
  tmp[3] = Base_Model_P.Gain_Gain_c * Attachment_Point_Vectors[8];
  tmp[6] = Attachment_Point_Vectors[7];
  tmp[1] = Attachment_Point_Vectors[8];
  tmp[4] = Base_Model_P.Constant_Value_i;
  _mm_storeu_pd(&tmp_a[0], _mm_mul_pd(_mm_set_pd(Base_Model_P.Gain2_Gain_m,
    Base_Model_P.Gain1_Gain_f), _mm_loadu_pd(&Attachment_Point_Vectors[6])));
  tmp[7] = tmp_a[0];
  tmp[2] = tmp_a[1];
  tmp[5] = Attachment_Point_Vectors[6];
  tmp[8] = Base_Model_P.Constant_Value_i;
  rtb_Sum_f_tmp_0[0] = Base_Model_P.Constant_Value_g4;
  rtb_Sum_f_tmp_0[3] = Base_Model_P.Gain_Gain_ii * Attachment_Point_Vectors[11];
  rtb_Sum_f_tmp_0[6] = Attachment_Point_Vectors[10];
  rtb_Sum_f_tmp_0[1] = Attachment_Point_Vectors[11];
  rtb_Sum_f_tmp_0[4] = Base_Model_P.Constant_Value_g4;
  _mm_storeu_pd(&tmp_a[0], _mm_mul_pd(_mm_set_pd(Base_Model_P.Gain2_Gain_oh,
    Base_Model_P.Gain1_Gain_j), _mm_loadu_pd(&Attachment_Point_Vectors[9])));
  rtb_Sum_f_tmp_0[7] = tmp_a[0];
  rtb_Sum_f_tmp_0[2] = tmp_a[1];
  rtb_Sum_f_tmp_0[5] = Attachment_Point_Vectors[9];
  rtb_Sum_f_tmp_0[8] = Base_Model_P.Constant_Value_g4;
  for (i = 0; i < 3; i++) {
    localProduct = Base_Model_U.Desired_Cable_Forces[i + 3];
    tmp_0[0] += tmp_1[3 * i] * localProduct;
    i_0 = 3 * i + 1;
    tmp_0[1] += tmp_1[i_0] * localProduct;
    tmp_d = 3 * i + 2;
    tmp_0[2] += tmp_1[tmp_d] * localProduct;
    localProduct = 0.0;
    Desired_Cable_Forces_Derivatives = 0.0;
    Desired_Cable_Forces_Derivatives_0 = 0.0;
    for (rtb_Transpose_tmp_0 = 0; rtb_Transpose_tmp_0 < 3; rtb_Transpose_tmp_0++)
    {
      rtb_Transpose_tmp = 3 * i + rtb_Transpose_tmp_0;
      rtb_Gain = rtb_Transpose[rtb_Transpose_tmp];
      tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&tmp[3 * rtb_Transpose_tmp_0]),
        _mm_set1_pd(rtb_Gain)), _mm_set_pd(Desired_Cable_Forces_Derivatives,
        localProduct));
      _mm_storeu_pd(&tmp_a[0], tmp_c);
      localProduct = tmp_a[0];
      Desired_Cable_Forces_Derivatives = tmp_a[1];
      Desired_Cable_Forces_Derivatives_0 += tmp[3 * rtb_Transpose_tmp_0 + 2] *
        rtb_Gain;
      tmp_2[rtb_Transpose_tmp] = 0.0;
    }

    rtb_Sum_f_tmp[tmp_d] = Desired_Cable_Forces_Derivatives_0;
    rtb_Sum_f_tmp[i_0] = Desired_Cable_Forces_Derivatives;
    rtb_Sum_f_tmp[3 * i] = localProduct;
    localProduct = tmp_2[3 * i];
    Desired_Cable_Forces_Derivatives = tmp_2[i_0];
    Desired_Cable_Forces_Derivatives_0 = tmp_2[tmp_d];
    for (rtb_Transpose_tmp_0 = 0; rtb_Transpose_tmp_0 < 3; rtb_Transpose_tmp_0++)
    {
      rtb_Gain = rtb_Transpose[3 * i + rtb_Transpose_tmp_0];
      tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Sum_f_tmp_0[3 *
        rtb_Transpose_tmp_0]), _mm_set1_pd(rtb_Gain)), _mm_set_pd
                         (Desired_Cable_Forces_Derivatives, localProduct));
      _mm_storeu_pd(&tmp_a[0], tmp_c);
      localProduct = tmp_a[0];
      Desired_Cable_Forces_Derivatives = tmp_a[1];
      Desired_Cable_Forces_Derivatives_0 += rtb_Sum_f_tmp_0[3 *
        rtb_Transpose_tmp_0 + 2] * rtb_Gain;
    }

    tmp_2[tmp_d] = Desired_Cable_Forces_Derivatives_0;
    tmp_2[i_0] = Desired_Cable_Forces_Derivatives;
    tmp_2[3 * i] = localProduct;
  }

  localProduct = Base_Model_U.Desired_Cable_Forces[7];
  Desired_Cable_Forces_Derivatives = Base_Model_U.Desired_Cable_Forces[6];
  Desired_Cable_Forces_Derivatives_0 = Base_Model_U.Desired_Cable_Forces[8];
  for (i = 0; i <= 0; i += 2) {
    tmp_c = _mm_loadu_pd(&rtb_Sum_f_tmp[i + 3]);
    tmp_b = _mm_loadu_pd(&rtb_Sum_f_tmp[i]);
    tmp_8 = _mm_loadu_pd(&rtb_Sum_f_tmp[i + 6]);
    tmp_9 = _mm_loadu_pd(&rtb_Sum_f_tmp_1[i]);
    tmp_7 = _mm_loadu_pd(&tmp_0[i]);
    _mm_storeu_pd(&tmp_3[i], _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(tmp_c,
      _mm_set1_pd(localProduct)), _mm_mul_pd(tmp_b, _mm_set1_pd
      (Desired_Cable_Forces_Derivatives))), _mm_mul_pd(tmp_8, _mm_set1_pd
      (Desired_Cable_Forces_Derivatives_0))), _mm_add_pd(tmp_9, tmp_7)));
    _mm_storeu_pd(&rtb_Force_Sum[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    tmp_3[i] = ((rtb_Sum_f_tmp[i + 3] * localProduct + rtb_Sum_f_tmp[i] *
                 Desired_Cable_Forces_Derivatives) + rtb_Sum_f_tmp[i + 6] *
                Desired_Cable_Forces_Derivatives_0) + (rtb_Sum_f_tmp_1[i] +
      tmp_0[i]);
  }

  rtb_MatrixMultiply7[0] = Base_Model_P.Gain1_Gain_fl *
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0];
  rtb_MatrixMultiply5_idx_0 = 0.0;
  rtb_MatrixMultiply7[1] = Base_Model_P.Gain1_Gain_fl *
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[1];
  rtb_MatrixMultiply5_idx_1 = 0.0;
  rtb_MatrixMultiply7[2] = Base_Model_P.Gain1_Gain_fl *
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2];
  rtb_MatrixMultiply5_idx_2 = 0.0;
  localProduct = 0.0;
  Desired_Cable_Forces_Derivatives = 0.0;
  Desired_Cable_Forces_Derivatives_0 = 0.0;
  for (i = 0; i < 3; i++) {
    rtb_Gain = Base_Model_U.Desired_Cable_Forces[i + 9];
    tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&tmp_2[3 * i]), _mm_set1_pd
      (rtb_Gain)), _mm_set_pd(Desired_Cable_Forces_Derivatives, localProduct));
    _mm_storeu_pd(&tmp_a[0], tmp_c);
    localProduct = tmp_a[0];
    Desired_Cable_Forces_Derivatives = tmp_a[1];
    _mm_storeu_pd(&tmp_a[0], _mm_add_pd(_mm_mul_pd(_mm_set_pd
      (Load_Inertia_Matrix[3 * i], tmp_2[3 * i + 2]), _mm_set_pd
      (Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[i], rtb_Gain)), _mm_set_pd
      (rtb_MatrixMultiply5_idx_0, Desired_Cable_Forces_Derivatives_0)));
    Desired_Cable_Forces_Derivatives_0 = tmp_a[0];
    rtb_MatrixMultiply5_idx_0 = tmp_a[1];
    _mm_storeu_pd(&tmp_a[0], _mm_add_pd(_mm_mul_pd(_mm_loadu_pd
      (&Load_Inertia_Matrix[3 * i + 1]), _mm_set1_pd
      (Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[i])), _mm_set_pd
      (rtb_MatrixMultiply5_idx_2, rtb_MatrixMultiply5_idx_1)));
    rtb_MatrixMultiply5_idx_1 = tmp_a[0];
    rtb_MatrixMultiply5_idx_2 = tmp_a[1];
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
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[0] +=
    Base_Model_P.DiscreteTimeIntegrator1_gainval *
    Base_Model_DW.DiscreteTimeIntegrator_DSTATE[0];
  Base_Model_DW.DiscreteTimeIntegrator_DSTATE[0] +=
    (((((Base_Model_U.Desired_Cable_Forces[0] +
         Base_Model_U.Desired_Cable_Forces[3]) +
        Base_Model_U.Desired_Cable_Forces[6]) +
       Base_Model_U.Desired_Cable_Forces[9]) + rtb_Gain *
      Base_Model_P.Constant_Value_fy[0]) -
     Base_Model_DW.DiscreteTimeIntegrator_DSTATE[0] * Load_Damping_Coef) /
    Load_Mass * Base_Model_P.DiscreteTimeIntegrator_gainval;
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[1] +=
    Base_Model_P.DiscreteTimeIntegrator1_gainval *
    Base_Model_DW.DiscreteTimeIntegrator_DSTATE[1];
  Base_Model_DW.DiscreteTimeIntegrator_DSTATE[1] +=
    (((((Base_Model_U.Desired_Cable_Forces[1] +
         Base_Model_U.Desired_Cable_Forces[4]) +
        Base_Model_U.Desired_Cable_Forces[7]) +
       Base_Model_U.Desired_Cable_Forces[10]) + rtb_Gain *
      Base_Model_P.Constant_Value_fy[1]) -
     Base_Model_DW.DiscreteTimeIntegrator_DSTATE[1] * Load_Damping_Coef) /
    Load_Mass * Base_Model_P.DiscreteTimeIntegrator_gainval;
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[2] +=
    Base_Model_P.DiscreteTimeIntegrator1_gainval *
    Base_Model_DW.DiscreteTimeIntegrator_DSTATE[2];
  Base_Model_DW.DiscreteTimeIntegrator_DSTATE[2] +=
    (((((Base_Model_U.Desired_Cable_Forces[2] +
         Base_Model_U.Desired_Cable_Forces[5]) +
        Base_Model_U.Desired_Cable_Forces[8]) +
       Base_Model_U.Desired_Cable_Forces[11]) + rtb_Gain *
      Base_Model_P.Constant_Value_fy[2]) -
     Base_Model_DW.DiscreteTimeIntegrator_DSTATE[2] * Load_Damping_Coef) /
    Load_Mass * Base_Model_P.DiscreteTimeIntegrator_gainval;
  for (i = 0; i <= 6; i += 2) {
    tmp_c = _mm_loadu_pd(&rtb_RL_dot[i]);
    tmp_b = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i]);
    _mm_storeu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i], _mm_add_pd
                  (_mm_mul_pd(_mm_set1_pd
      (Base_Model_P.DiscreteTimeIntegrator2_gainval), tmp_c), tmp_b));
  }

  for (i = 8; i < 9; i++) {
    Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i] +=
      Base_Model_P.DiscreteTimeIntegrator2_gainval * rtb_RL_dot[i];
  }

  rt_invd3x3_snf(Load_Inertia_Matrix, tmp);
  rtb_Force_Sum[0] = ((rtb_MatrixMultiply7[1] * rtb_MatrixMultiply5_idx_2 -
                       rtb_MatrixMultiply5_idx_1 * rtb_MatrixMultiply7[2]) +
                      (tmp_3[0] + localProduct)) -
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0] * Load_Damping_Coef;
  localProduct = 0.0;
  rtb_Force_Sum[1] = ((rtb_MatrixMultiply5_idx_0 * rtb_MatrixMultiply7[2] -
                       rtb_MatrixMultiply7[0] * rtb_MatrixMultiply5_idx_2) +
                      (tmp_3[1] + Desired_Cable_Forces_Derivatives)) -
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[1] * Load_Damping_Coef;
  Desired_Cable_Forces_Derivatives = 0.0;
  rtb_Force_Sum[2] = ((rtb_MatrixMultiply7[0] * rtb_MatrixMultiply5_idx_1 -
                       rtb_MatrixMultiply5_idx_0 * rtb_MatrixMultiply7[1]) +
                      (tmp_3[2] + Desired_Cable_Forces_Derivatives_0)) -
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2] * Load_Damping_Coef;
  Desired_Cable_Forces_Derivatives_0 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_c = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&tmp[3 * i]), _mm_set1_pd
      (rtb_Force_Sum[i])), _mm_set_pd(Desired_Cable_Forces_Derivatives,
      localProduct));
    _mm_storeu_pd(&tmp_a[0], tmp_c);
    localProduct = tmp_a[0];
    Desired_Cable_Forces_Derivatives = tmp_a[1];
    Desired_Cable_Forces_Derivatives_0 += tmp[3 * i + 2] * rtb_Force_Sum[i];
  }

  tmp_c = _mm_set_pd(Base_Model_P.DiscreteTimeIntegrator1_gainval_g,
                     Base_Model_P.DiscreteTimeIntegrator3_gainval);
  _mm_storeu_pd(&tmp_a[0], _mm_add_pd(_mm_mul_pd(tmp_c, _mm_set_pd
    (Base_Model_DW.UnitDelay_DSTATE[0], localProduct)), _mm_set_pd
    (Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[0],
     Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0])));
  Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0] = tmp_a[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[0] = tmp_a[1];
  tmp_b = _mm_set_pd(Base_Model_P.DiscreteTimeIntegrator1_gainval_i,
                     Base_Model_P.DiscreteTimeIntegrator1_gainval_n);
  _mm_storeu_pd(&tmp_a[0], _mm_add_pd(_mm_mul_pd(tmp_b, _mm_set_pd
    (Base_Model_DW.UnitDelay_DSTATE_k[0], Base_Model_DW.UnitDelay_DSTATE_b[0])),
    _mm_set_pd(Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[0],
               Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[0])));
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[0] = tmp_a[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[0] = tmp_a[1];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[0] +=
    Base_Model_P.DiscreteTimeIntegrator1_gainval_g0 *
    Base_Model_DW.UnitDelay_DSTATE_c[0];
  _mm_storeu_pd(&tmp_a[0], _mm_add_pd(_mm_set_pd(rtb_Force_Sum_0[0],
    rtb_Transpose1[0]), _mm_set_pd(rtb_Sum_f_tmp_5, rtb_RL_dot_0)));
  Base_Model_DW.UnitDelay_DSTATE[0] = tmp_a[0];
  Base_Model_DW.UnitDelay_DSTATE_b[0] = tmp_a[1];
  Base_Model_DW.UnitDelay_DSTATE_k[0] = rtb_MatrixMultiply7_0[0] +
    rtb_Sum_f_tmp_2;
  Base_Model_DW.UnitDelay_DSTATE_c[0] = rtb_Sum_p[0];
  _mm_storeu_pd(&tmp_a[0], _mm_add_pd(_mm_mul_pd(tmp_c, _mm_set_pd
    (Base_Model_DW.UnitDelay_DSTATE[1], Desired_Cable_Forces_Derivatives)),
    _mm_set_pd(Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[1],
               Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[1])));
  Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[1] = tmp_a[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[1] = tmp_a[1];
  _mm_storeu_pd(&tmp_a[0], _mm_add_pd(_mm_mul_pd(tmp_b, _mm_set_pd
    (Base_Model_DW.UnitDelay_DSTATE_k[1], Base_Model_DW.UnitDelay_DSTATE_b[1])),
    _mm_set_pd(Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[1],
               Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[1])));
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[1] = tmp_a[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[1] = tmp_a[1];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[1] +=
    Base_Model_P.DiscreteTimeIntegrator1_gainval_g0 *
    Base_Model_DW.UnitDelay_DSTATE_c[1];
  _mm_storeu_pd(&tmp_a[0], _mm_add_pd(_mm_set_pd(rtb_Force_Sum_0[1],
    rtb_Transpose1[1]), _mm_set_pd(rtb_Sum_f_tmp_6, rtb_RL_dot_1)));
  Base_Model_DW.UnitDelay_DSTATE[1] = tmp_a[0];
  Base_Model_DW.UnitDelay_DSTATE_b[1] = tmp_a[1];
  Base_Model_DW.UnitDelay_DSTATE_k[1] = rtb_MatrixMultiply7_0[1] +
    rtb_Sum_f_tmp_3;
  Base_Model_DW.UnitDelay_DSTATE_c[1] = rtb_Sum_p[1];
  _mm_storeu_pd(&tmp_a[0], _mm_add_pd(_mm_mul_pd(tmp_c, _mm_set_pd
    (Base_Model_DW.UnitDelay_DSTATE[2], Desired_Cable_Forces_Derivatives_0)),
    _mm_set_pd(Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[2],
               Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2])));
  Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2] = tmp_a[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[2] = tmp_a[1];
  _mm_storeu_pd(&tmp_a[0], _mm_add_pd(_mm_mul_pd(tmp_b, _mm_set_pd
    (Base_Model_DW.UnitDelay_DSTATE_k[2], Base_Model_DW.UnitDelay_DSTATE_b[2])),
    _mm_set_pd(Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[2],
               Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[2])));
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[2] = tmp_a[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[2] = tmp_a[1];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[2] +=
    Base_Model_P.DiscreteTimeIntegrator1_gainval_g0 *
    Base_Model_DW.UnitDelay_DSTATE_c[2];
  _mm_storeu_pd(&tmp_a[0], _mm_add_pd(_mm_set_pd(rtb_Force_Sum_0[2],
    rtb_Transpose1[2]), _mm_set_pd(rtb_Sum_f_tmp_7, rtb_RL_dot_2)));
  Base_Model_DW.UnitDelay_DSTATE[2] = tmp_a[0];
  Base_Model_DW.UnitDelay_DSTATE_b[2] = tmp_a[1];
  Base_Model_DW.UnitDelay_DSTATE_k[2] = rtb_MatrixMultiply7_0[2] +
    rtb_Sum_f_tmp_4;
  Base_Model_DW.UnitDelay_DSTATE_c[2] = rtb_Sum_p[2];
  fmu_LogOutput();
}

void Base_Model_initialize(void)
{
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[0] =
    Base_Model_P.DiscreteTimeIntegrator1_IC[0];
  Base_Model_DW.DiscreteTimeIntegrator_DSTATE[0] =
    Base_Model_P.DiscreteTimeIntegrator_IC;
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[1] =
    Base_Model_P.DiscreteTimeIntegrator1_IC[1];
  Base_Model_DW.DiscreteTimeIntegrator_DSTATE[1] =
    Base_Model_P.DiscreteTimeIntegrator_IC;
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[2] =
    Base_Model_P.DiscreteTimeIntegrator1_IC[2];
  Base_Model_DW.DiscreteTimeIntegrator_DSTATE[2] =
    Base_Model_P.DiscreteTimeIntegrator_IC;
  memcpy(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[0],
         &Base_Model_P.DiscreteTimeIntegrator2_IC[0], 9U * sizeof(real_T));
  Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0] =
    Base_Model_P.DiscreteTimeIntegrator3_IC;
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[0] =
    Base_Model_P.DiscreteTimeIntegrator1_IC_h[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[0] =
    Base_Model_P.DiscreteTimeIntegrator1_IC_e[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[0] =
    Base_Model_P.DiscreteTimeIntegrator1_IC_n[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[0] =
    Base_Model_P.DiscreteTimeIntegrator1_IC_b[0];
  Base_Model_DW.UnitDelay_DSTATE[0] = Base_Model_P.UnitDelay_InitialCondition;
  Base_Model_DW.UnitDelay_DSTATE_b[0] =
    Base_Model_P.UnitDelay_InitialCondition_b;
  Base_Model_DW.UnitDelay_DSTATE_k[0] =
    Base_Model_P.UnitDelay_InitialCondition_o;
  Base_Model_DW.UnitDelay_DSTATE_c[0] =
    Base_Model_P.UnitDelay_InitialCondition_c;
  Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[1] =
    Base_Model_P.DiscreteTimeIntegrator3_IC;
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[1] =
    Base_Model_P.DiscreteTimeIntegrator1_IC_h[1];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[1] =
    Base_Model_P.DiscreteTimeIntegrator1_IC_e[1];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[1] =
    Base_Model_P.DiscreteTimeIntegrator1_IC_n[1];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[1] =
    Base_Model_P.DiscreteTimeIntegrator1_IC_b[1];
  Base_Model_DW.UnitDelay_DSTATE[1] = Base_Model_P.UnitDelay_InitialCondition;
  Base_Model_DW.UnitDelay_DSTATE_b[1] =
    Base_Model_P.UnitDelay_InitialCondition_b;
  Base_Model_DW.UnitDelay_DSTATE_k[1] =
    Base_Model_P.UnitDelay_InitialCondition_o;
  Base_Model_DW.UnitDelay_DSTATE_c[1] =
    Base_Model_P.UnitDelay_InitialCondition_c;
  Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2] =
    Base_Model_P.DiscreteTimeIntegrator3_IC;
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[2] =
    Base_Model_P.DiscreteTimeIntegrator1_IC_h[2];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[2] =
    Base_Model_P.DiscreteTimeIntegrator1_IC_e[2];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[2] =
    Base_Model_P.DiscreteTimeIntegrator1_IC_n[2];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[2] =
    Base_Model_P.DiscreteTimeIntegrator1_IC_b[2];
  Base_Model_DW.UnitDelay_DSTATE[2] = Base_Model_P.UnitDelay_InitialCondition;
  Base_Model_DW.UnitDelay_DSTATE_b[2] =
    Base_Model_P.UnitDelay_InitialCondition_b;
  Base_Model_DW.UnitDelay_DSTATE_k[2] =
    Base_Model_P.UnitDelay_InitialCondition_o;
  Base_Model_DW.UnitDelay_DSTATE_c[2] =
    Base_Model_P.UnitDelay_InitialCondition_c;
}

void Base_Model_terminate(void)
{
}

RT_MODEL_Base_Model_T *Base_Model(void)
{
  (void) memset((void *)Base_Model_M, 0,
                sizeof(RT_MODEL_Base_Model_T));
  (void) memset((void *)&Base_Model_DW, 0,
                sizeof(DW_Base_Model_T));
  (void)memset(&Base_Model_U, 0, sizeof(ExtU_Base_Model_T));
  (void)memset(&Base_Model_Y, 0, sizeof(ExtY_Base_Model_T));
  return Base_Model_M;
}
