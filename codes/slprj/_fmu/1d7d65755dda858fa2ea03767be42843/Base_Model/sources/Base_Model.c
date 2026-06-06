#include "Base_Model_macros.h"
#include "Base_Model.h"
#include "rtwtypes.h"
#include <string.h>
#include "rt_nonfinite.h"
#include <emmintrin.h>
#include <math.h>
#include "rt_assert.h"
#include "Base_Model_private.h"

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
static real_T Base_Model_xzlangeM(const real_T x[9]);
static void Base_Model_xzlascl(real_T cfrom, real_T cto, real_T A[9]);
static real_T Base_Model_xnrm2(int32_T n, const real_T x[9], int32_T ix0);
static real_T Base_Model_xdotc(int32_T n, const real_T x[9], int32_T ix0, const
  real_T y[9], int32_T iy0);
static void Base_Model_xaxpy(int32_T n, real_T a, int32_T ix0, real_T y[9],
  int32_T iy0);
static real_T Base_Model_xnrm2_a(const real_T x[3], int32_T ix0);
static void Base_Model_xaxpy_a(int32_T n, real_T a, const real_T x[9], int32_T
  ix0, real_T y[3], int32_T iy0);
static void Base_Model_xaxpy_a3(int32_T n, real_T a, const real_T x[3], int32_T
  ix0, real_T y[9], int32_T iy0);
static void Base_Model_xzlascl_e(real_T cfrom, real_T cto, real_T A[3]);
static void Base_Model_xswap(real_T x[9], int32_T ix0, int32_T iy0);
static void Base_Model_xrotg(real_T *a, real_T *b, real_T *c, real_T *s);
static void Base_Model_xrot(real_T x[9], int32_T ix0, int32_T iy0, real_T c,
  real_T s);
static void Base_Model_svd(const real_T A[9], real_T U[9], real_T s[3], real_T
  V[9]);
static real_T Base_Model_xzlangeM(const real_T x[9])
{
  real_T y;
  int32_T k;
  boolean_T exitg1;
  y = 0.0;
  k = 0;
  exitg1 = false;
  while ((!exitg1) && (k < 9)) {
    real_T absxk;
    absxk = fabs(x[k]);
    if (rtIsNaN(absxk)) {
      y = (rtNaN);
      exitg1 = true;
    } else {
      if (absxk > y) {
        y = absxk;
      }

      k++;
    }
  }

  return y;
}

static void Base_Model_xzlascl(real_T cfrom, real_T cto, real_T A[9])
{
  real_T cfromc;
  real_T ctoc;
  int32_T j;
  boolean_T notdone;
  cfromc = cfrom;
  ctoc = cto;
  notdone = true;
  while (notdone) {
    real_T cfrom1;
    real_T cto1;
    real_T mul;
    cfrom1 = cfromc * 2.0041683600089728E-292;
    cto1 = ctoc / 4.9896007738368E+291;
    if ((fabs(cfrom1) > fabs(ctoc)) && (ctoc != 0.0)) {
      mul = 2.0041683600089728E-292;
      cfromc = cfrom1;
    } else if (fabs(cto1) > fabs(cfromc)) {
      mul = 4.9896007738368E+291;
      ctoc = cto1;
    } else {
      mul = ctoc / cfromc;
      notdone = false;
    }

    for (j = 0; j < 3; j++) {
      int32_T tmp;
      A[j * 3] *= mul;
      tmp = j * 3 + 1;
      A[tmp] *= mul;
      tmp = j * 3 + 2;
      A[tmp] *= mul;
    }
  }
}

static real_T Base_Model_xnrm2(int32_T n, const real_T x[9], int32_T ix0)
{
  real_T scale;
  real_T y;
  int32_T k;
  int32_T kend;
  y = 0.0;
  scale = 3.3121686421112381E-170;
  kend = (ix0 + n) - 1;
  for (k = ix0; k <= kend; k++) {
    real_T absxk;
    absxk = fabs(x[k - 1]);
    if (absxk > scale) {
      real_T t;
      t = scale / absxk;
      y = y * t * t + 1.0;
      scale = absxk;
    } else {
      real_T t;
      t = absxk / scale;
      y += t * t;
    }
  }

  y = scale * sqrt(y);
  if (rtIsNaN(y)) {
    k = ix0;
    int32_T exitg1;
    do {
      exitg1 = 0;
      if (k <= kend) {
        if (rtIsNaN(x[k - 1])) {
          exitg1 = 1;
        } else {
          k++;
        }
      } else {
        y = (rtInf);
        exitg1 = 1;
      }
    } while (exitg1 == 0);
  }

  return y;
}

static real_T Base_Model_xdotc(int32_T n, const real_T x[9], int32_T ix0, const
  real_T y[9], int32_T iy0)
{
  real_T d;
  int32_T b;
  int32_T k;
  d = 0.0;
  b = (uint8_T)n;
  for (k = 0; k < b; k++) {
    d += x[(ix0 + k) - 1] * y[(iy0 + k) - 1];
  }

  return d;
}

static void Base_Model_xaxpy(int32_T n, real_T a, int32_T ix0, real_T y[9],
  int32_T iy0)
{
  int32_T k;
  if (!(a == 0.0)) {
    for (k = 0; k < n; k++) {
      int32_T tmp;
      tmp = (iy0 + k) - 1;
      y[tmp] += y[(ix0 + k) - 1] * a;
    }
  }
}

static real_T Base_Model_xnrm2_a(const real_T x[3], int32_T ix0)
{
  real_T scale;
  real_T y;
  int32_T k;
  y = 0.0;
  scale = 3.3121686421112381E-170;
  for (k = ix0; k <= ix0 + 1; k++) {
    real_T absxk;
    absxk = fabs(x[k - 1]);
    if (absxk > scale) {
      real_T t;
      t = scale / absxk;
      y = y * t * t + 1.0;
      scale = absxk;
    } else {
      real_T t;
      t = absxk / scale;
      y += t * t;
    }
  }

  y = scale * sqrt(y);
  if (rtIsNaN(y)) {
    k = ix0;
    int32_T exitg1;
    do {
      exitg1 = 0;
      if (k <= ix0 + 1) {
        if (rtIsNaN(x[k - 1])) {
          exitg1 = 1;
        } else {
          k++;
        }
      } else {
        y = (rtInf);
        exitg1 = 1;
      }
    } while (exitg1 == 0);
  }

  return y;
}

static void Base_Model_xaxpy_a(int32_T n, real_T a, const real_T x[9], int32_T
  ix0, real_T y[3], int32_T iy0)
{
  int32_T k;
  if (!(a == 0.0)) {
    int32_T scalarLB;
    int32_T tmp_0;
    int32_T vectorUB;
    scalarLB = (n / 2) << 1;
    vectorUB = scalarLB - 2;
    for (k = 0; k <= vectorUB; k += 2) {
      __m128d tmp;
      tmp_0 = (iy0 + k) - 1;
      tmp = _mm_loadu_pd(&y[tmp_0]);
      _mm_storeu_pd(&y[tmp_0], _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&x[(ix0 + k) -
        1]), _mm_set1_pd(a)), tmp));
    }

    for (k = scalarLB; k < n; k++) {
      tmp_0 = (iy0 + k) - 1;
      y[tmp_0] += x[(ix0 + k) - 1] * a;
    }
  }
}

static void Base_Model_xaxpy_a3(int32_T n, real_T a, const real_T x[3], int32_T
  ix0, real_T y[9], int32_T iy0)
{
  int32_T k;
  if (!(a == 0.0)) {
    int32_T scalarLB;
    int32_T tmp_0;
    int32_T vectorUB;
    scalarLB = (n / 2) << 1;
    vectorUB = scalarLB - 2;
    for (k = 0; k <= vectorUB; k += 2) {
      __m128d tmp;
      tmp_0 = (iy0 + k) - 1;
      tmp = _mm_loadu_pd(&y[tmp_0]);
      _mm_storeu_pd(&y[tmp_0], _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&x[(ix0 + k) -
        1]), _mm_set1_pd(a)), tmp));
    }

    for (k = scalarLB; k < n; k++) {
      tmp_0 = (iy0 + k) - 1;
      y[tmp_0] += x[(ix0 + k) - 1] * a;
    }
  }
}

static void Base_Model_xzlascl_e(real_T cfrom, real_T cto, real_T A[3])
{
  real_T cfromc;
  real_T ctoc;
  boolean_T notdone;
  cfromc = cfrom;
  ctoc = cto;
  notdone = true;
  while (notdone) {
    __m128d tmp;
    real_T cfrom1;
    real_T cto1;
    real_T mul;
    cfrom1 = cfromc * 2.0041683600089728E-292;
    cto1 = ctoc / 4.9896007738368E+291;
    if ((fabs(cfrom1) > fabs(ctoc)) && (ctoc != 0.0)) {
      mul = 2.0041683600089728E-292;
      cfromc = cfrom1;
    } else if (fabs(cto1) > fabs(cfromc)) {
      mul = 4.9896007738368E+291;
      ctoc = cto1;
    } else {
      mul = ctoc / cfromc;
      notdone = false;
    }

    tmp = _mm_mul_pd(_mm_loadu_pd(&A[0]), _mm_set1_pd(mul));
    _mm_storeu_pd(&A[0], tmp);
    A[2] *= mul;
  }
}

static void Base_Model_xswap(real_T x[9], int32_T ix0, int32_T iy0)
{
  real_T temp;
  temp = x[ix0 - 1];
  x[ix0 - 1] = x[iy0 - 1];
  x[iy0 - 1] = temp;
  temp = x[ix0];
  x[ix0] = x[iy0];
  x[iy0] = temp;
  temp = x[ix0 + 1];
  x[ix0 + 1] = x[iy0 + 1];
  x[iy0 + 1] = temp;
}

static void Base_Model_xrotg(real_T *a, real_T *b, real_T *c, real_T *s)
{
  real_T absa;
  real_T absb;
  real_T roe;
  real_T scale;
  roe = *b;
  absa = fabs(*a);
  absb = fabs(*b);
  if (absa > absb) {
    roe = *a;
  }

  scale = absa + absb;
  if (scale == 0.0) {
    *s = 0.0;
    *c = 1.0;
    *a = 0.0;
    *b = 0.0;
  } else {
    real_T ads;
    real_T bds;
    ads = absa / scale;
    bds = absb / scale;
    scale *= sqrt(ads * ads + bds * bds);
    if (roe < 0.0) {
      scale = -scale;
    }

    *c = *a / scale;
    *s = *b / scale;
    if (absa > absb) {
      *b = *s;
    } else if (*c != 0.0) {
      *b = 1.0 / *c;
    } else {
      *b = 1.0;
    }

    *a = scale;
  }
}

static void Base_Model_xrot(real_T x[9], int32_T ix0, int32_T iy0, real_T c,
  real_T s)
{
  real_T temp;
  real_T temp_tmp;
  temp = x[iy0 - 1];
  temp_tmp = x[ix0 - 1];
  x[iy0 - 1] = temp * c - temp_tmp * s;
  x[ix0 - 1] = temp_tmp * c + temp * s;
  temp = x[ix0] * c + x[iy0] * s;
  x[iy0] = x[iy0] * c - x[ix0] * s;
  x[ix0] = temp;
  temp = x[iy0 + 1];
  temp_tmp = x[ix0 + 1];
  x[iy0 + 1] = temp * c - temp_tmp * s;
  x[ix0 + 1] = temp_tmp * c + temp * s;
}

static void Base_Model_svd(const real_T A[9], real_T U[9], real_T s[3], real_T
  V[9])
{
  __m128d tmp;
  real_T b_A[9];
  real_T b_s[3];
  real_T e[3];
  real_T work[3];
  real_T tmp_0[2];
  real_T anrm;
  real_T cscale;
  real_T nrm;
  real_T rt;
  real_T shift;
  real_T smm1;
  real_T sqds;
  real_T ztest;
  int32_T exitg1;
  int32_T kase;
  int32_T m;
  int32_T qjj;
  int32_T qp1;
  int32_T qq;
  int32_T qq_tmp;
  int32_T scalarLB;
  int32_T vectorUB;
  boolean_T apply_transform;
  boolean_T doscale;
  boolean_T exitg2;
  b_s[0] = 0.0;
  e[0] = 0.0;
  work[0] = 0.0;
  b_s[1] = 0.0;
  e[1] = 0.0;
  work[1] = 0.0;
  b_s[2] = 0.0;
  e[2] = 0.0;
  work[2] = 0.0;
  for (kase = 0; kase < 9; kase++) {
    b_A[kase] = A[kase];
    U[kase] = 0.0;
    V[kase] = 0.0;
  }

  doscale = false;
  anrm = Base_Model_xzlangeM(A);
  cscale = anrm;
  if ((anrm > 0.0) && (anrm < 6.7178761075670888E-139)) {
    doscale = true;
    cscale = 6.7178761075670888E-139;
    Base_Model_xzlascl(anrm, cscale, b_A);
  } else if (anrm > 1.4885657073574029E+138) {
    doscale = true;
    cscale = 1.4885657073574029E+138;
    Base_Model_xzlascl(anrm, cscale, b_A);
  }

  for (m = 0; m < 2; m++) {
    qp1 = m + 2;
    qq_tmp = 3 * m + m;
    qq = qq_tmp + 1;
    apply_transform = false;
    nrm = Base_Model_xnrm2(3 - m, b_A, qq_tmp + 1);
    if (nrm > 0.0) {
      apply_transform = true;
      if (b_A[qq_tmp] < 0.0) {
        nrm = -nrm;
      }

      b_s[m] = nrm;
      if (fabs(nrm) >= 1.0020841800044864E-292) {
        nrm = 1.0 / nrm;
        qjj = (qq_tmp - m) + 3;
        scalarLB = ((((qjj - qq_tmp) / 2) << 1) + qq_tmp) + 1;
        vectorUB = scalarLB - 2;
        for (kase = qq; kase <= vectorUB; kase += 2) {
          tmp = _mm_loadu_pd(&b_A[kase - 1]);
          _mm_storeu_pd(&b_A[kase - 1], _mm_mul_pd(tmp, _mm_set1_pd(nrm)));
        }

        for (kase = scalarLB; kase <= qjj; kase++) {
          b_A[kase - 1] *= nrm;
        }
      } else {
        qjj = (qq_tmp - m) + 3;
        scalarLB = ((((qjj - qq_tmp) / 2) << 1) + qq_tmp) + 1;
        vectorUB = scalarLB - 2;
        for (kase = qq; kase <= vectorUB; kase += 2) {
          tmp = _mm_loadu_pd(&b_A[kase - 1]);
          _mm_storeu_pd(&b_A[kase - 1], _mm_div_pd(tmp, _mm_set1_pd(b_s[m])));
        }

        for (kase = scalarLB; kase <= qjj; kase++) {
          b_A[kase - 1] /= b_s[m];
        }
      }

      b_A[qq_tmp]++;
      b_s[m] = -b_s[m];
    } else {
      b_s[m] = 0.0;
    }

    for (kase = qp1; kase < 4; kase++) {
      qjj = (kase - 1) * 3 + m;
      if (apply_transform) {
        Base_Model_xaxpy(3 - m, -(Base_Model_xdotc(3 - m, b_A, qq_tmp + 1, b_A,
          qjj + 1) / b_A[qq_tmp]), qq_tmp + 1, b_A, qjj + 1);
      }

      e[kase - 1] = b_A[qjj];
    }

    for (qq = m + 1; qq < 4; qq++) {
      kase = (3 * m + qq) - 1;
      U[kase] = b_A[kase];
    }

    if (m <= 0) {
      nrm = Base_Model_xnrm2_a(e, 2);
      if (nrm == 0.0) {
        e[0] = 0.0;
      } else {
        if (e[1] < 0.0) {
          e[0] = -nrm;
        } else {
          e[0] = nrm;
        }

        nrm = e[0];
        if (fabs(e[0]) >= 1.0020841800044864E-292) {
          nrm = 1.0 / e[0];
          scalarLB = ((((2 - m) / 2) << 1) + m) + 2;
          vectorUB = scalarLB - 2;
          for (qq = qp1; qq <= vectorUB; qq += 2) {
            tmp = _mm_loadu_pd(&e[qq - 1]);
            _mm_storeu_pd(&e[qq - 1], _mm_mul_pd(tmp, _mm_set1_pd(nrm)));
          }

          for (qq = scalarLB; qq < 4; qq++) {
            e[qq - 1] *= nrm;
          }
        } else {
          scalarLB = ((((2 - m) / 2) << 1) + m) + 2;
          vectorUB = scalarLB - 2;
          for (qq = qp1; qq <= vectorUB; qq += 2) {
            tmp = _mm_loadu_pd(&e[qq - 1]);
            _mm_storeu_pd(&e[qq - 1], _mm_div_pd(tmp, _mm_set1_pd(nrm)));
          }

          for (qq = scalarLB; qq < 4; qq++) {
            e[qq - 1] /= nrm;
          }
        }

        e[1]++;
        e[0] = -e[0];
        for (qq = qp1; qq < 4; qq++) {
          work[qq - 1] = 0.0;
        }

        for (qq = qp1; qq < 4; qq++) {
          Base_Model_xaxpy_a(2, e[qq - 1], b_A, 3 * (qq - 1) + 2, work, 2);
        }

        for (qq = qp1; qq < 4; qq++) {
          Base_Model_xaxpy_a3(2, -e[qq - 1] / e[1], work, 2, b_A, 3 * (qq - 1) +
                              2);
        }
      }

      for (qq = qp1; qq < 4; qq++) {
        V[qq - 1] = e[qq - 1];
      }
    }
  }

  m = 1;
  b_s[2] = b_A[8];
  e[1] = b_A[7];
  e[2] = 0.0;
  U[6] = 0.0;
  U[7] = 0.0;
  U[8] = 1.0;
  for (qp1 = 1; qp1 >= 0; qp1--) {
    qq = 3 * qp1 + qp1;
    if (b_s[qp1] != 0.0) {
      for (kase = qp1 + 2; kase < 4; kase++) {
        qjj = ((kase - 1) * 3 + qp1) + 1;
        Base_Model_xaxpy(3 - qp1, -(Base_Model_xdotc(3 - qp1, U, qq + 1, U, qjj)
          / U[qq]), qq + 1, U, qjj);
      }

      for (qjj = qp1 + 1; qjj < 4; qjj++) {
        kase = (3 * qp1 + qjj) - 1;
        U[kase] = -U[kase];
      }

      U[qq]++;
      if (qp1 - 1 >= 0) {
        U[3 * qp1] = 0.0;
      }
    } else {
      U[3 * qp1] = 0.0;
      U[3 * qp1 + 1] = 0.0;
      U[3 * qp1 + 2] = 0.0;
      U[qq] = 1.0;
    }
  }

  for (qp1 = 2; qp1 >= 0; qp1--) {
    if ((qp1 <= 0) && (e[0] != 0.0)) {
      Base_Model_xaxpy(2, -(Base_Model_xdotc(2, V, 2, V, 5) / V[1]), 2, V, 5);
      Base_Model_xaxpy(2, -(Base_Model_xdotc(2, V, 2, V, 8) / V[1]), 2, V, 8);
    }

    V[3 * qp1] = 0.0;
    V[3 * qp1 + 1] = 0.0;
    V[3 * qp1 + 2] = 0.0;
    V[qp1 + 3 * qp1] = 1.0;
  }

  for (qp1 = 0; qp1 < 3; qp1++) {
    nrm = b_s[qp1];
    if (nrm != 0.0) {
      rt = fabs(nrm);
      nrm /= rt;
      b_s[qp1] = rt;
      if (qp1 + 1 < 3) {
        e[qp1] /= nrm;
      }

      qq = 3 * qp1 + 1;
      scalarLB = 2 + qq;
      for (qjj = qq; qjj <= qq; qjj += 2) {
        tmp = _mm_loadu_pd(&U[qjj - 1]);
        _mm_storeu_pd(&U[qjj - 1], _mm_mul_pd(tmp, _mm_set1_pd(nrm)));
      }

      for (qjj = scalarLB; qjj <= qq + 2; qjj++) {
        U[qjj - 1] *= nrm;
      }
    }

    if (qp1 + 1 < 3) {
      smm1 = e[qp1];
      if (smm1 != 0.0) {
        rt = fabs(smm1);
        nrm = rt / smm1;
        e[qp1] = rt;
        b_s[qp1 + 1] *= nrm;
        qq = (qp1 + 1) * 3 + 1;
        scalarLB = 2 + qq;
        for (qjj = qq; qjj <= qq; qjj += 2) {
          tmp = _mm_loadu_pd(&V[qjj - 1]);
          _mm_storeu_pd(&V[qjj - 1], _mm_mul_pd(tmp, _mm_set1_pd(nrm)));
        }

        for (qjj = scalarLB; qjj <= qq + 2; qjj++) {
          V[qjj - 1] *= nrm;
        }
      }
    }
  }

  qp1 = 0;
  nrm = fmax(fmax(fmax(0.0, fmax(fabs(b_s[0]), fabs(e[0]))), fmax(fabs(b_s[1]),
    fabs(e[1]))), fmax(fabs(b_s[2]), fabs(e[2])));
  while ((m + 2 > 0) && (qp1 < 75)) {
    kase = m + 1;
    do {
      exitg1 = 0;
      qq = kase;
      if (kase == 0) {
        exitg1 = 1;
      } else {
        rt = fabs(e[kase - 1]);
        if ((rt <= (fabs(b_s[kase - 1]) + fabs(b_s[kase])) *
             2.2204460492503131E-16) || ((rt <= 1.0020841800044864E-292) ||
             ((qp1 > 20) && (rt <= 2.2204460492503131E-16 * nrm)))) {
          e[kase - 1] = 0.0;
          exitg1 = 1;
        } else {
          kase--;
        }
      }
    } while (exitg1 == 0);

    if (m + 1 == kase) {
      kase = 4;
    } else {
      qjj = m + 2;
      qq_tmp = m + 2;
      exitg2 = false;
      while ((!exitg2) && (qq_tmp >= kase)) {
        qjj = qq_tmp;
        if (qq_tmp == kase) {
          exitg2 = true;
        } else {
          rt = 0.0;
          if (qq_tmp < m + 2) {
            rt = fabs(e[qq_tmp - 1]);
          }

          if (qq_tmp > kase + 1) {
            rt += fabs(e[qq_tmp - 2]);
          }

          ztest = fabs(b_s[qq_tmp - 1]);
          if ((ztest <= 2.2204460492503131E-16 * rt) || (ztest <=
               1.0020841800044864E-292)) {
            b_s[qq_tmp - 1] = 0.0;
            exitg2 = true;
          } else {
            qq_tmp--;
          }
        }
      }

      if (qjj == kase) {
        kase = 3;
      } else if (m + 2 == qjj) {
        kase = 1;
      } else {
        kase = 2;
        qq = qjj;
      }
    }

    switch (kase) {
     case 1:
      rt = e[m];
      e[m] = 0.0;
      for (qjj = m + 1; qjj >= qq + 1; qjj--) {
        Base_Model_xrotg(&b_s[qjj - 1], &rt, &ztest, &sqds);
        if (qjj > qq + 1) {
          rt = -sqds * e[0];
          e[0] *= ztest;
        }

        Base_Model_xrot(V, 3 * (qjj - 1) + 1, 3 * (m + 1) + 1, ztest, sqds);
      }
      break;

     case 2:
      rt = e[qq - 1];
      e[qq - 1] = 0.0;
      for (qjj = qq + 1; qjj <= m + 2; qjj++) {
        Base_Model_xrotg(&b_s[qjj - 1], &rt, &ztest, &sqds);
        smm1 = e[qjj - 1];
        rt = -sqds * smm1;
        e[qjj - 1] = smm1 * ztest;
        Base_Model_xrot(U, 3 * (qjj - 1) + 1, 3 * (qq - 1) + 1, ztest, sqds);
      }
      break;

     case 3:
      rt = b_s[m + 1];
      ztest = fmax(fmax(fmax(fmax(fabs(rt), fabs(b_s[m])), fabs(e[m])), fabs
                        (b_s[qq])), fabs(e[qq]));
      tmp = _mm_set1_pd(ztest);
      _mm_storeu_pd(&tmp_0[0], _mm_div_pd(_mm_set_pd(b_s[m], rt), tmp));
      rt = tmp_0[0];
      smm1 = tmp_0[1];
      _mm_storeu_pd(&tmp_0[0], _mm_div_pd(_mm_set_pd(b_s[qq], e[m]), tmp));
      smm1 = ((smm1 + rt) * (smm1 - rt) + tmp_0[0] * tmp_0[0]) / 2.0;
      sqds = rt * tmp_0[0];
      sqds *= sqds;
      if ((smm1 != 0.0) || (sqds != 0.0)) {
        shift = sqrt(smm1 * smm1 + sqds);
        if (smm1 < 0.0) {
          shift = -shift;
        }

        shift = sqds / (smm1 + shift);
      } else {
        shift = 0.0;
      }

      rt = (tmp_0[1] + rt) * (tmp_0[1] - rt) + shift;
      ztest = e[qq] / ztest * tmp_0[1];
      for (qjj = qq + 1; qjj <= m + 1; qjj++) {
        Base_Model_xrotg(&rt, &ztest, &sqds, &smm1);
        if (qjj > qq + 1) {
          e[0] = rt;
        }

        shift = e[qjj - 1];
        rt = b_s[qjj - 1];
        e[qjj - 1] = shift * sqds - rt * smm1;
        ztest = smm1 * b_s[qjj];
        b_s[qjj] *= sqds;
        kase = (qjj - 1) * 3 + 1;
        qq_tmp = 3 * qjj + 1;
        Base_Model_xrot(V, kase, qq_tmp, sqds, smm1);
        b_s[qjj - 1] = rt * sqds + shift * smm1;
        Base_Model_xrotg(&b_s[qjj - 1], &ztest, &sqds, &smm1);
        shift = e[qjj - 1];
        rt = shift * sqds + smm1 * b_s[qjj];
        b_s[qjj] = shift * -smm1 + sqds * b_s[qjj];
        ztest = smm1 * e[qjj];
        e[qjj] *= sqds;
        Base_Model_xrot(U, kase, qq_tmp, sqds, smm1);
      }

      e[m] = rt;
      qp1++;
      break;

     default:
      if (b_s[qq] < 0.0) {
        b_s[qq] = -b_s[qq];
        qp1 = 3 * qq + 1;
        scalarLB = 2 + qp1;
        for (qjj = qp1; qjj <= qp1; qjj += 2) {
          tmp = _mm_loadu_pd(&V[qjj - 1]);
          _mm_storeu_pd(&V[qjj - 1], _mm_mul_pd(tmp, _mm_set1_pd(-1.0)));
        }

        for (qjj = scalarLB; qjj <= qp1 + 2; qjj++) {
          V[qjj - 1] = -V[qjj - 1];
        }
      }

      qp1 = qq + 1;
      while ((qq + 1 < 3) && (b_s[qq] < b_s[qp1])) {
        rt = b_s[qq];
        b_s[qq] = b_s[qp1];
        b_s[qp1] = rt;
        kase = 3 * qq + 1;
        qq_tmp = (qq + 1) * 3 + 1;
        Base_Model_xswap(V, kase, qq_tmp);
        Base_Model_xswap(U, kase, qq_tmp);
        qq = qp1;
        qp1++;
      }

      qp1 = 0;
      m--;
      break;
    }
  }

  s[0] = b_s[0];
  s[1] = b_s[1];
  s[2] = b_s[2];
  if (doscale) {
    Base_Model_xzlascl_e(cscale, anrm, s);
  }
}

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
  __m128d tmp_3;
  __m128d tmp_4;
  __m128d tmp_5;
  __m128d tmp_6;
  __m128d tmp_7;
  __m128d tmp_8;
  __m128d tmp_a;
  __m128d tmp_b;
  real_T A[9];
  real_T U[9];
  real_T rtb_R_clean[9];
  real_T rtb_Sum_f_tmp[9];
  real_T rtb_Transpose[9];
  real_T tmp[9];
  real_T tmp_1[9];
  real_T rtb_Force_Sum[3];
  real_T rtb_Force_Sum_0[3];
  real_T rtb_MatrixMultiply7[3];
  real_T rtb_MatrixMultiply7_0[3];
  real_T rtb_Sum_p[3];
  real_T rtb_Transpose1[3];
  real_T s[3];
  real_T tmp_0[3];
  real_T tmp_2[3];
  real_T tmp_9[2];
  real_T Attachment_Point_Vectors_0;
  real_T Attachment_Point_Vectors_1;
  real_T Desired_Cable_Forces_Derivatives;
  real_T U_0;
  real_T b_s;
  real_T rtb_MatrixMultiply5_idx_0;
  real_T rtb_MatrixMultiply5_idx_1;
  real_T rtb_MatrixMultiply5_idx_2;
  real_T rtb_R_clean_0;
  real_T rtb_R_clean_1;
  real_T rtb_R_clean_2;
  real_T rtb_Sum_f_tmp_0;
  real_T rtb_Sum_f_tmp_1;
  real_T rtb_Sum_f_tmp_2;
  real_T rtb_Sum_f_tmp_3;
  real_T rtb_Sum_f_tmp_4;
  real_T rtb_Sum_f_tmp_5;
  real_T smax;
  int32_T b_ix;
  int32_T d;
  int32_T d_k;
  int32_T i;
  int32_T ijA;
  int32_T ix;
  int32_T iy;
  int32_T jj;
  int8_T ipiv[3];
  boolean_T isodd;
  Base_Model_Y.Load_Position[0] = Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[0];
  Base_Model_Y.Load_LinVelocity[0] =
    Base_Model_DW.DiscreteTimeIntegrator_DSTATE[0];
  Base_Model_Y.Load_Position[1] = Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[1];
  Base_Model_Y.Load_LinVelocity[1] =
    Base_Model_DW.DiscreteTimeIntegrator_DSTATE[1];
  Base_Model_Y.Load_Position[2] = Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[2];
  Base_Model_Y.Load_LinVelocity[2] =
    Base_Model_DW.DiscreteTimeIntegrator_DSTATE[2];
  isodd = true;
  for (i = 0; i < 9; i++) {
    if (isodd && ((!rtIsInf(Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i])) &&
                  (!rtIsNaN(Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i]))))
    {
    } else {
      isodd = false;
    }
  }

  if (isodd) {
    Base_Model_svd(Base_Model_DW.DiscreteTimeIntegrator2_DSTATE, U, s,
                   rtb_R_clean);
  } else {
    for (i = 0; i < 9; i++) {
      U[i] = (rtNaN);
      rtb_R_clean[i] = (rtNaN);
    }
  }

  for (i = 0; i < 3; i++) {
    rtb_Transpose[3 * i] = rtb_R_clean[i];
    rtb_Transpose[3 * i + 1] = rtb_R_clean[i + 3];
    rtb_Transpose[3 * i + 2] = rtb_R_clean[i + 6];
  }

  for (i = 0; i < 3; i++) {
    rtb_R_clean_0 = 0.0;
    rtb_R_clean_1 = 0.0;
    rtb_R_clean_2 = 0.0;
    for (jj = 0; jj < 3; jj++) {
      b_s = rtb_Transpose[3 * i + jj];
      tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&U[3 * jj]), _mm_set1_pd(b_s)),
                         _mm_set_pd(rtb_R_clean_1, rtb_R_clean_0));
      _mm_storeu_pd(&tmp_9[0], tmp_b);
      rtb_R_clean_0 = tmp_9[0];
      rtb_R_clean_1 = tmp_9[1];
      rtb_R_clean_2 += U[3 * jj + 2] * b_s;
    }

    rtb_R_clean[3 * i + 2] = rtb_R_clean_2;
    rtb_R_clean[3 * i + 1] = rtb_R_clean_1;
    rtb_R_clean[3 * i] = rtb_R_clean_0;
  }

  memcpy(&A[0], &rtb_R_clean[0], 9U * sizeof(real_T));
  ipiv[0] = 1;
  ipiv[1] = 2;
  for (i = 0; i < 2; i++) {
    jj = i << 2;
    iy = 4 - i;
    b_ix = 0;
    ix = jj;
    smax = fabs(A[jj]);
    for (d_k = 2; d_k < iy; d_k++) {
      ix++;
      b_s = fabs(A[ix]);
      if (b_s > smax) {
        b_ix = d_k - 1;
        smax = b_s;
      }
    }

    if (A[jj + b_ix] != 0.0) {
      if (b_ix != 0) {
        iy = i + b_ix;
        ipiv[i] = (int8_T)(iy + 1);
        smax = A[i];
        A[i] = A[iy];
        A[iy] = smax;
        smax = A[i + 3];
        A[i + 3] = A[iy + 3];
        A[iy + 3] = smax;
        smax = A[i + 6];
        A[i + 6] = A[iy + 6];
        A[iy + 6] = smax;
      }

      iy = (jj - i) + 3;
      ix = (((((iy - jj) - 1) / 2) << 1) + jj) + 2;
      d_k = ix - 2;
      for (b_ix = jj + 2; b_ix <= d_k; b_ix += 2) {
        tmp_b = _mm_loadu_pd(&A[b_ix - 1]);
        _mm_storeu_pd(&A[b_ix - 1], _mm_div_pd(tmp_b, _mm_set1_pd(A[jj])));
      }

      for (b_ix = ix; b_ix <= iy; b_ix++) {
        A[b_ix - 1] /= A[jj];
      }
    }

    iy = jj + 3;
    b_ix = jj + 5;
    ix = 1 - i;
    for (d_k = 0; d_k <= ix; d_k++) {
      smax = A[iy];
      if (A[iy] != 0.0) {
        d = (b_ix - i) + 1;
        for (ijA = b_ix; ijA <= d; ijA++) {
          A[ijA - 1] += A[((jj + ijA) - b_ix) + 1] * -smax;
        }
      }

      iy += 3;
      b_ix += 3;
    }
  }

  isodd = (ipiv[0] > 1);
  smax = A[0] * A[4] * A[8];
  if (ipiv[1] > 2) {
    isodd = !isodd;
  }

  if (isodd) {
    smax = -smax;
  }

  if (smax < 0.0) {
    for (i = 0; i < 3; i++) {
      U[i + 6] = -U[i + 6];
      rtb_R_clean[3 * i] = 0.0;
      rtb_R_clean[3 * i + 1] = 0.0;
      rtb_R_clean[3 * i + 2] = 0.0;
    }

    for (i = 0; i < 3; i++) {
      rtb_R_clean_0 = rtb_R_clean[3 * i];
      iy = 3 * i + 1;
      rtb_R_clean_1 = rtb_R_clean[iy];
      b_ix = 3 * i + 2;
      rtb_R_clean_2 = rtb_R_clean[b_ix];
      for (jj = 0; jj < 3; jj++) {
        b_s = rtb_Transpose[3 * i + jj];
        tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&U[3 * jj]), _mm_set1_pd(b_s)),
                           _mm_set_pd(rtb_R_clean_1, rtb_R_clean_0));
        _mm_storeu_pd(&tmp_9[0], tmp_b);
        rtb_R_clean_0 = tmp_9[0];
        rtb_R_clean_1 = tmp_9[1];
        rtb_R_clean_2 += U[3 * jj + 2] * b_s;
      }

      rtb_R_clean[b_ix] = rtb_R_clean_2;
      rtb_R_clean[iy] = rtb_R_clean_1;
      rtb_R_clean[3 * i] = rtb_R_clean_0;
    }
  }

  memcpy(&Base_Model_Y.Load_Orientation[0], &rtb_R_clean[0], 9U * sizeof(real_T));
  tmp[0] = Base_Model_P.Constant_Value;
  tmp[3] = Base_Model_P.Gain_Gain *
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2];
  tmp[6] = Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[1];
  tmp[1] = Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2];
  tmp[4] = Base_Model_P.Constant_Value;
  tmp_b = _mm_mul_pd(_mm_set_pd(Base_Model_P.Gain2_Gain, Base_Model_P.Gain1_Gain),
                     _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator3_DSTATE
    [0]));
  _mm_storeu_pd(&tmp_9[0], tmp_b);
  tmp[7] = tmp_9[0];
  tmp[2] = tmp_9[1];
  tmp[5] = Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0];
  tmp[8] = Base_Model_P.Constant_Value;
  for (i = 0; i < 3; i++) {
    rtb_R_clean_1 = 0.0;
    rtb_R_clean_2 = 0.0;
    U_0 = 0.0;
    for (jj = 0; jj < 3; jj++) {
      b_s = tmp[3 * i + jj];
      tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_R_clean[3 * jj]),
        _mm_set1_pd(b_s)), _mm_set_pd(rtb_R_clean_2, rtb_R_clean_1));
      _mm_storeu_pd(&tmp_9[0], tmp_b);
      rtb_R_clean_1 = tmp_9[0];
      rtb_R_clean_2 = tmp_9[1];
      U_0 += rtb_R_clean[3 * jj + 2] * b_s;
    }

    U[3 * i + 2] = U_0;
    U[3 * i + 1] = rtb_R_clean_2;
    U[3 * i] = rtb_R_clean_1;
  }

  memcpy(&Base_Model_Y.Load_Orientation_Derivative[0], &U[0], 9U * sizeof(real_T));
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
  b_s = sqrt((Base_Model_U.Desired_Cable_Forces[0] *
              Base_Model_U.Desired_Cable_Forces[0] +
              Base_Model_U.Desired_Cable_Forces[1] *
              Base_Model_U.Desired_Cable_Forces[1]) +
             Base_Model_U.Desired_Cable_Forces[2] *
             Base_Model_U.Desired_Cable_Forces[2]);
  if (b_s > Base_Model_P.Saturation_UpperSat) {
    b_s = Base_Model_P.Saturation_UpperSat;
  } else if (b_s < Base_Model_P.Saturation_LowerSat) {
    b_s = Base_Model_P.Saturation_LowerSat;
  }

  tmp_b = _mm_set1_pd(b_s);
  _mm_storeu_pd(&rtb_Transpose1[0], _mm_div_pd(_mm_loadu_pd
    (&Base_Model_U.Desired_Cable_Forces[0]), tmp_b));
  rtb_Transpose1[2] = Base_Model_U.Desired_Cable_Forces[2] / b_s;
  rtb_Transpose[0] = Base_Model_P.Constant_Value_g;
  rtb_Transpose[3] = Base_Model_P.Gain_Gain_n *
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2];
  rtb_Transpose[6] = Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[1];
  rtb_Transpose[1] = Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2];
  rtb_Transpose[4] = Base_Model_P.Constant_Value_g;
  tmp_a = _mm_mul_pd(_mm_set_pd(Base_Model_P.Gain2_Gain_o,
    Base_Model_P.Gain1_Gain_d), _mm_loadu_pd
                     (&Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0]));
  _mm_storeu_pd(&tmp_9[0], tmp_a);
  rtb_Transpose[7] = tmp_9[0];
  rtb_Transpose[2] = tmp_9[1];
  rtb_Transpose[5] = Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0];
  rtb_Transpose[8] = Base_Model_P.Constant_Value_g;
  for (i = 0; i < 3; i++) {
    tmp_a = _mm_sub_pd(_mm_loadu_pd(&Base_Model_P.Constant_Value_m[3 * i]),
                       _mm_mul_pd(_mm_loadu_pd(&rtb_Transpose1[0]), _mm_set1_pd
      (rtb_Transpose1[i])));
    _mm_storeu_pd(&tmp[3 * i], tmp_a);
    jj = 3 * i + 2;
    tmp[jj] = Base_Model_P.Constant_Value_m[jj] - rtb_Transpose1[2] *
      rtb_Transpose1[i];
  }

  Desired_Cable_Forces_Derivatives =
    Base_Model_U.Desired_Cable_Forces_Derivatives[1];
  rtb_R_clean_0 = Base_Model_U.Desired_Cable_Forces_Derivatives[0];
  rtb_MatrixMultiply5_idx_0 = Base_Model_U.Desired_Cable_Forces_Derivatives[2];
  for (i = 0; i <= 0; i += 2) {
    tmp_a = _mm_loadu_pd(&tmp[i + 3]);
    tmp_7 = _mm_loadu_pd(&tmp[i]);
    tmp_8 = _mm_loadu_pd(&tmp[i + 6]);
    _mm_storeu_pd(&s[i], _mm_div_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(tmp_a,
      _mm_set1_pd(Desired_Cable_Forces_Derivatives)), _mm_mul_pd(tmp_7,
      _mm_set1_pd(rtb_R_clean_0))), _mm_mul_pd(tmp_8, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_0))), tmp_b));
    _mm_storeu_pd(&rtb_MatrixMultiply7[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    s[i] = ((tmp[i + 3] * Desired_Cable_Forces_Derivatives + tmp[i] *
             rtb_R_clean_0) + tmp[i + 6] * rtb_MatrixMultiply5_idx_0) / b_s;
  }

  rtb_MatrixMultiply5_idx_0 = 0.0;
  rtb_MatrixMultiply5_idx_1 = 0.0;
  rtb_MatrixMultiply5_idx_2 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Transpose[3 * i]),
      _mm_set1_pd(Attachment_Point_Vectors[i])), _mm_set_pd
                       (rtb_MatrixMultiply5_idx_1, rtb_MatrixMultiply5_idx_0));
    _mm_storeu_pd(&tmp_9[0], tmp_b);
    rtb_MatrixMultiply5_idx_0 = tmp_9[0];
    rtb_MatrixMultiply5_idx_1 = tmp_9[1];
    rtb_MatrixMultiply5_idx_2 += rtb_Transpose[3 * i + 2] *
      Attachment_Point_Vectors[i];
  }

  for (i = 0; i <= 0; i += 2) {
    tmp_b = _mm_loadu_pd(&rtb_R_clean[i + 3]);
    tmp_a = _mm_loadu_pd(&rtb_R_clean[i]);
    tmp_7 = _mm_loadu_pd(&rtb_R_clean[i + 6]);
    tmp_8 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]);
    tmp_6 = _mm_loadu_pd(&s[i]);
    _mm_storeu_pd(&rtb_Force_Sum[i], _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_b, _mm_set1_pd(rtb_MatrixMultiply5_idx_1)), _mm_mul_pd
       (tmp_a, _mm_set1_pd(rtb_MatrixMultiply5_idx_0))), _mm_mul_pd(tmp_7,
      _mm_set1_pd(rtb_MatrixMultiply5_idx_2))), tmp_8), _mm_mul_pd(_mm_set1_pd
      (Cable_Resting_Length), tmp_6)));
  }

  for (i = 2; i < 3; i++) {
    rtb_Force_Sum[i] = (((rtb_R_clean[i + 3] * rtb_MatrixMultiply5_idx_1 +
                          rtb_R_clean[i] * rtb_MatrixMultiply5_idx_0) +
                         rtb_R_clean[i + 6] * rtb_MatrixMultiply5_idx_2) +
                        Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]) +
      Cable_Resting_Length * s[i];
  }

  for (i = 0; i <= 6; i += 2) {
    tmp_b = _mm_loadu_pd(&Base_Model_ConstP.pooled1[i]);
    _mm_storeu_pd(&A[i], _mm_mul_pd(_mm_set1_pd(Kpc), tmp_b));
    _mm_storeu_pd(&rtb_Sum_f_tmp[i], _mm_mul_pd(_mm_set1_pd(Kdc), tmp_b));
  }

  for (i = 8; i < 9; i++) {
    rtb_R_clean_1 = Base_Model_ConstP.pooled1[i];
    A[i] = Kpc * rtb_R_clean_1;
    rtb_Sum_f_tmp[i] = Kdc * rtb_R_clean_1;
  }

  smax = Attachment_Point_Vectors[1];
  Attachment_Point_Vectors_0 = Attachment_Point_Vectors[0];
  Attachment_Point_Vectors_1 = Attachment_Point_Vectors[2];
  for (i = 0; i <= 0; i += 2) {
    tmp_b = _mm_loadu_pd(&rtb_R_clean[i + 3]);
    tmp_a = _mm_loadu_pd(&rtb_R_clean[i]);
    tmp_7 = _mm_loadu_pd(&rtb_R_clean[i + 6]);
    tmp_8 = _mm_loadu_pd(&rtb_Transpose1[i]);
    tmp_6 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]);
    tmp_5 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[i]);
    _mm_storeu_pd(&s[i], _mm_sub_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_b, _mm_set1_pd(smax)), _mm_mul_pd(tmp_a, _mm_set1_pd
      (Attachment_Point_Vectors_0))), _mm_mul_pd(tmp_7, _mm_set1_pd
      (Attachment_Point_Vectors_1))), _mm_mul_pd(tmp_8, _mm_set1_pd
      (Cable_Resting_Length))), tmp_6), tmp_5));
    tmp_b = _mm_loadu_pd(&rtb_Force_Sum[i]);
    tmp_a = _mm_loadu_pd(&Base_Model_DW.UnitDelay_DSTATE[i]);
    _mm_storeu_pd(&rtb_Force_Sum_0[i], _mm_sub_pd(tmp_b, tmp_a));
  }

  for (i = 2; i < 3; i++) {
    s[i] = ((((rtb_R_clean[i + 3] * smax + rtb_R_clean[i] *
               Attachment_Point_Vectors_0) + rtb_R_clean[i + 6] *
              Attachment_Point_Vectors_1) + rtb_Transpose1[i] *
             Cable_Resting_Length) +
            Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]) -
      Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[i];
    rtb_Force_Sum_0[i] = rtb_Force_Sum[i] - Base_Model_DW.UnitDelay_DSTATE[i];
  }

  b_s = s[1];
  Desired_Cable_Forces_Derivatives = s[0];
  rtb_R_clean_0 = s[2];
  for (i = 0; i <= 0; i += 2) {
    tmp_b = _mm_loadu_pd(&A[i + 3]);
    tmp_a = _mm_loadu_pd(&A[i]);
    tmp_7 = _mm_loadu_pd(&A[i + 6]);
    tmp_8 = _mm_loadu_pd(&rtb_Force_Sum[i]);
    _mm_storeu_pd(&rtb_Transpose1[i], _mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_b, _mm_set1_pd(b_s)), _mm_mul_pd(tmp_a, _mm_set1_pd
      (Desired_Cable_Forces_Derivatives))), _mm_mul_pd(tmp_7, _mm_set1_pd
      (rtb_R_clean_0))), tmp_8));
    _mm_storeu_pd(&s[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    rtb_Transpose1[i] = ((A[i + 3] * b_s + A[i] *
                          Desired_Cable_Forces_Derivatives) + A[i + 6] *
                         rtb_R_clean_0) + rtb_Force_Sum[i];
  }

  rtb_R_clean_1 = 0.0;
  rtb_R_clean_2 = 0.0;
  U_0 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Sum_f_tmp[3 * i]),
      _mm_set1_pd(rtb_Force_Sum_0[i])), _mm_set_pd(rtb_R_clean_2, rtb_R_clean_1));
    _mm_storeu_pd(&tmp_9[0], tmp_b);
    rtb_R_clean_1 = tmp_9[0];
    rtb_R_clean_2 = tmp_9[1];
    U_0 += rtb_Sum_f_tmp[3 * i + 2] * rtb_Force_Sum_0[i];
  }

  b_s = sqrt((Base_Model_U.Desired_Cable_Forces[3] *
              Base_Model_U.Desired_Cable_Forces[3] +
              Base_Model_U.Desired_Cable_Forces[4] *
              Base_Model_U.Desired_Cable_Forces[4]) +
             Base_Model_U.Desired_Cable_Forces[5] *
             Base_Model_U.Desired_Cable_Forces[5]);
  if (b_s > Base_Model_P.Saturation1_UpperSat) {
    b_s = Base_Model_P.Saturation1_UpperSat;
  } else if (b_s < Base_Model_P.Saturation1_LowerSat) {
    b_s = Base_Model_P.Saturation1_LowerSat;
  }

  tmp_b = _mm_set1_pd(b_s);
  _mm_storeu_pd(&rtb_Force_Sum[0], _mm_div_pd(_mm_loadu_pd
    (&Base_Model_U.Desired_Cable_Forces[3]), tmp_b));
  rtb_Force_Sum[2] = Base_Model_U.Desired_Cable_Forces[5] / b_s;
  for (i = 0; i < 3; i++) {
    tmp_a = _mm_sub_pd(_mm_loadu_pd(&Base_Model_P.Constant1_Value[3 * i]),
                       _mm_mul_pd(_mm_loadu_pd(&rtb_Force_Sum[0]), _mm_set1_pd
      (rtb_Force_Sum[i])));
    _mm_storeu_pd(&tmp[3 * i], tmp_a);
    jj = 3 * i + 2;
    tmp[jj] = Base_Model_P.Constant1_Value[jj] - rtb_Force_Sum[2] *
      rtb_Force_Sum[i];
  }

  Desired_Cable_Forces_Derivatives =
    Base_Model_U.Desired_Cable_Forces_Derivatives[4];
  rtb_R_clean_0 = Base_Model_U.Desired_Cable_Forces_Derivatives[3];
  rtb_MatrixMultiply5_idx_0 = Base_Model_U.Desired_Cable_Forces_Derivatives[5];
  for (i = 0; i <= 0; i += 2) {
    tmp_a = _mm_loadu_pd(&tmp[i + 3]);
    tmp_7 = _mm_loadu_pd(&tmp[i]);
    tmp_8 = _mm_loadu_pd(&tmp[i + 6]);
    _mm_storeu_pd(&s[i], _mm_div_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(tmp_a,
      _mm_set1_pd(Desired_Cable_Forces_Derivatives)), _mm_mul_pd(tmp_7,
      _mm_set1_pd(rtb_R_clean_0))), _mm_mul_pd(tmp_8, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_0))), tmp_b));
    _mm_storeu_pd(&rtb_MatrixMultiply7[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    s[i] = ((tmp[i + 3] * Desired_Cable_Forces_Derivatives + tmp[i] *
             rtb_R_clean_0) + tmp[i + 6] * rtb_MatrixMultiply5_idx_0) / b_s;
  }

  rtb_MatrixMultiply5_idx_0 = 0.0;
  rtb_MatrixMultiply5_idx_1 = 0.0;
  rtb_MatrixMultiply5_idx_2 = 0.0;
  for (i = 0; i < 3; i++) {
    b_s = Attachment_Point_Vectors[i + 3];
    tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Transpose[3 * i]),
      _mm_set1_pd(b_s)), _mm_set_pd(rtb_MatrixMultiply5_idx_1,
      rtb_MatrixMultiply5_idx_0));
    _mm_storeu_pd(&tmp_9[0], tmp_b);
    rtb_MatrixMultiply5_idx_0 = tmp_9[0];
    rtb_MatrixMultiply5_idx_1 = tmp_9[1];
    rtb_MatrixMultiply5_idx_2 += rtb_Transpose[3 * i + 2] * b_s;
  }

  smax = Attachment_Point_Vectors[3];
  Attachment_Point_Vectors_0 = Attachment_Point_Vectors[4];
  Attachment_Point_Vectors_1 = Attachment_Point_Vectors[5];
  for (i = 0; i <= 0; i += 2) {
    tmp_b = _mm_loadu_pd(&rtb_R_clean[i]);
    tmp_a = _mm_loadu_pd(&rtb_R_clean[i + 3]);
    tmp_7 = _mm_loadu_pd(&rtb_R_clean[i + 6]);
    tmp_8 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]);
    tmp_6 = _mm_loadu_pd(&s[i]);
    tmp_5 = _mm_set1_pd(Cable_Resting_Length);
    tmp_8 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(tmp_7, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_2)), _mm_add_pd(_mm_mul_pd(tmp_a, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_1)), _mm_mul_pd(tmp_b, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_0)))), tmp_8), _mm_mul_pd(tmp_5, tmp_6));
    _mm_storeu_pd(&rtb_MatrixMultiply7[i], tmp_8);
    tmp_6 = _mm_loadu_pd(&rtb_Force_Sum[i]);
    tmp_3 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]);
    tmp_4 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[i]);
    _mm_storeu_pd(&tmp_0[i], _mm_sub_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_7, _mm_set1_pd(Attachment_Point_Vectors_1)), _mm_add_pd
       (_mm_mul_pd(tmp_a, _mm_set1_pd(Attachment_Point_Vectors_0)), _mm_mul_pd
        (tmp_b, _mm_set1_pd(smax)))), _mm_mul_pd(tmp_6, tmp_5)), tmp_3), tmp_4));
    tmp_b = _mm_loadu_pd(&Base_Model_DW.UnitDelay_DSTATE_b[i]);
    _mm_storeu_pd(&rtb_Sum_p[i], _mm_sub_pd(tmp_8, tmp_b));
  }

  for (i = 2; i < 3; i++) {
    rtb_R_clean_0 = rtb_R_clean[i];
    b_s = rtb_R_clean_0 * rtb_MatrixMultiply5_idx_0;
    Desired_Cable_Forces_Derivatives = rtb_R_clean_0 * smax;
    rtb_R_clean_0 = rtb_R_clean[i + 3];
    b_s += rtb_R_clean_0 * rtb_MatrixMultiply5_idx_1;
    Desired_Cable_Forces_Derivatives += rtb_R_clean_0 *
      Attachment_Point_Vectors_0;
    rtb_R_clean_0 = rtb_R_clean[i + 6];
    b_s = ((rtb_R_clean_0 * rtb_MatrixMultiply5_idx_2 + b_s) +
           Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]) +
      Cable_Resting_Length * s[i];
    rtb_MatrixMultiply7[i] = b_s;
    tmp_0[i] = (((rtb_R_clean_0 * Attachment_Point_Vectors_1 +
                  Desired_Cable_Forces_Derivatives) + rtb_Force_Sum[i] *
                 Cable_Resting_Length) +
                Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]) -
      Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[i];
    rtb_Sum_p[i] = b_s - Base_Model_DW.UnitDelay_DSTATE_b[i];
  }

  b_s = tmp_0[1];
  Desired_Cable_Forces_Derivatives = tmp_0[0];
  rtb_R_clean_0 = tmp_0[2];
  for (i = 0; i <= 0; i += 2) {
    tmp_b = _mm_loadu_pd(&A[i + 3]);
    tmp_a = _mm_loadu_pd(&A[i]);
    tmp_7 = _mm_loadu_pd(&A[i + 6]);
    tmp_8 = _mm_loadu_pd(&rtb_MatrixMultiply7[i]);
    _mm_storeu_pd(&rtb_Force_Sum_0[i], _mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_b, _mm_set1_pd(b_s)), _mm_mul_pd(tmp_a, _mm_set1_pd
      (Desired_Cable_Forces_Derivatives))), _mm_mul_pd(tmp_7, _mm_set1_pd
      (rtb_R_clean_0))), tmp_8));
    _mm_storeu_pd(&s[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    rtb_Force_Sum_0[i] = ((A[i + 3] * b_s + A[i] *
      Desired_Cable_Forces_Derivatives) + A[i + 6] * rtb_R_clean_0) +
      rtb_MatrixMultiply7[i];
  }

  rtb_Sum_f_tmp_3 = 0.0;
  rtb_Sum_f_tmp_4 = 0.0;
  rtb_Sum_f_tmp_5 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Sum_f_tmp[3 * i]),
      _mm_set1_pd(rtb_Sum_p[i])), _mm_set_pd(rtb_Sum_f_tmp_4, rtb_Sum_f_tmp_3));
    _mm_storeu_pd(&tmp_9[0], tmp_b);
    rtb_Sum_f_tmp_3 = tmp_9[0];
    rtb_Sum_f_tmp_4 = tmp_9[1];
    rtb_Sum_f_tmp_5 += rtb_Sum_f_tmp[3 * i + 2] * rtb_Sum_p[i];
  }

  b_s = sqrt((Base_Model_U.Desired_Cable_Forces[6] *
              Base_Model_U.Desired_Cable_Forces[6] +
              Base_Model_U.Desired_Cable_Forces[7] *
              Base_Model_U.Desired_Cable_Forces[7]) +
             Base_Model_U.Desired_Cable_Forces[8] *
             Base_Model_U.Desired_Cable_Forces[8]);
  if (b_s > Base_Model_P.Saturation2_UpperSat) {
    b_s = Base_Model_P.Saturation2_UpperSat;
  } else if (b_s < Base_Model_P.Saturation2_LowerSat) {
    b_s = Base_Model_P.Saturation2_LowerSat;
  }

  tmp_b = _mm_set1_pd(b_s);
  _mm_storeu_pd(&rtb_Force_Sum[0], _mm_div_pd(_mm_loadu_pd
    (&Base_Model_U.Desired_Cable_Forces[6]), tmp_b));
  rtb_Force_Sum[2] = Base_Model_U.Desired_Cable_Forces[8] / b_s;
  for (i = 0; i < 3; i++) {
    tmp_a = _mm_sub_pd(_mm_loadu_pd(&Base_Model_P.Constant2_Value[3 * i]),
                       _mm_mul_pd(_mm_loadu_pd(&rtb_Force_Sum[0]), _mm_set1_pd
      (rtb_Force_Sum[i])));
    _mm_storeu_pd(&tmp[3 * i], tmp_a);
    jj = 3 * i + 2;
    tmp[jj] = Base_Model_P.Constant2_Value[jj] - rtb_Force_Sum[2] *
      rtb_Force_Sum[i];
  }

  Desired_Cable_Forces_Derivatives =
    Base_Model_U.Desired_Cable_Forces_Derivatives[7];
  rtb_R_clean_0 = Base_Model_U.Desired_Cable_Forces_Derivatives[6];
  rtb_MatrixMultiply5_idx_0 = Base_Model_U.Desired_Cable_Forces_Derivatives[8];
  for (i = 0; i <= 0; i += 2) {
    tmp_a = _mm_loadu_pd(&tmp[i + 3]);
    tmp_7 = _mm_loadu_pd(&tmp[i]);
    tmp_8 = _mm_loadu_pd(&tmp[i + 6]);
    _mm_storeu_pd(&s[i], _mm_div_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(tmp_a,
      _mm_set1_pd(Desired_Cable_Forces_Derivatives)), _mm_mul_pd(tmp_7,
      _mm_set1_pd(rtb_R_clean_0))), _mm_mul_pd(tmp_8, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_0))), tmp_b));
    _mm_storeu_pd(&rtb_MatrixMultiply7[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    s[i] = ((tmp[i + 3] * Desired_Cable_Forces_Derivatives + tmp[i] *
             rtb_R_clean_0) + tmp[i + 6] * rtb_MatrixMultiply5_idx_0) / b_s;
  }

  rtb_MatrixMultiply5_idx_0 = 0.0;
  rtb_MatrixMultiply5_idx_1 = 0.0;
  rtb_MatrixMultiply5_idx_2 = 0.0;
  for (i = 0; i < 3; i++) {
    b_s = Attachment_Point_Vectors[i + 6];
    tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Transpose[3 * i]),
      _mm_set1_pd(b_s)), _mm_set_pd(rtb_MatrixMultiply5_idx_1,
      rtb_MatrixMultiply5_idx_0));
    _mm_storeu_pd(&tmp_9[0], tmp_b);
    rtb_MatrixMultiply5_idx_0 = tmp_9[0];
    rtb_MatrixMultiply5_idx_1 = tmp_9[1];
    rtb_MatrixMultiply5_idx_2 += rtb_Transpose[3 * i + 2] * b_s;
  }

  smax = Attachment_Point_Vectors[6];
  Attachment_Point_Vectors_0 = Attachment_Point_Vectors[7];
  Attachment_Point_Vectors_1 = Attachment_Point_Vectors[8];
  for (i = 0; i <= 0; i += 2) {
    tmp_b = _mm_loadu_pd(&rtb_R_clean[i]);
    tmp_a = _mm_loadu_pd(&rtb_R_clean[i + 3]);
    tmp_7 = _mm_loadu_pd(&rtb_R_clean[i + 6]);
    tmp_8 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]);
    tmp_6 = _mm_loadu_pd(&s[i]);
    tmp_5 = _mm_set1_pd(Cable_Resting_Length);
    tmp_8 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(tmp_7, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_2)), _mm_add_pd(_mm_mul_pd(tmp_a, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_1)), _mm_mul_pd(tmp_b, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_0)))), tmp_8), _mm_mul_pd(tmp_5, tmp_6));
    _mm_storeu_pd(&rtb_MatrixMultiply7[i], tmp_8);
    tmp_6 = _mm_loadu_pd(&rtb_Force_Sum[i]);
    tmp_3 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]);
    tmp_4 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[i]);
    _mm_storeu_pd(&tmp_0[i], _mm_sub_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_7, _mm_set1_pd(Attachment_Point_Vectors_1)), _mm_add_pd
       (_mm_mul_pd(tmp_a, _mm_set1_pd(Attachment_Point_Vectors_0)), _mm_mul_pd
        (tmp_b, _mm_set1_pd(smax)))), _mm_mul_pd(tmp_6, tmp_5)), tmp_3), tmp_4));
    tmp_b = _mm_loadu_pd(&Base_Model_DW.UnitDelay_DSTATE_k[i]);
    _mm_storeu_pd(&rtb_Sum_p[i], _mm_sub_pd(tmp_8, tmp_b));
  }

  for (i = 2; i < 3; i++) {
    rtb_R_clean_0 = rtb_R_clean[i];
    b_s = rtb_R_clean_0 * rtb_MatrixMultiply5_idx_0;
    Desired_Cable_Forces_Derivatives = rtb_R_clean_0 * smax;
    rtb_R_clean_0 = rtb_R_clean[i + 3];
    b_s += rtb_R_clean_0 * rtb_MatrixMultiply5_idx_1;
    Desired_Cable_Forces_Derivatives += rtb_R_clean_0 *
      Attachment_Point_Vectors_0;
    rtb_R_clean_0 = rtb_R_clean[i + 6];
    b_s = ((rtb_R_clean_0 * rtb_MatrixMultiply5_idx_2 + b_s) +
           Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]) +
      Cable_Resting_Length * s[i];
    rtb_MatrixMultiply7[i] = b_s;
    tmp_0[i] = (((rtb_R_clean_0 * Attachment_Point_Vectors_1 +
                  Desired_Cable_Forces_Derivatives) + rtb_Force_Sum[i] *
                 Cable_Resting_Length) +
                Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]) -
      Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[i];
    rtb_Sum_p[i] = b_s - Base_Model_DW.UnitDelay_DSTATE_k[i];
  }

  b_s = tmp_0[1];
  Desired_Cable_Forces_Derivatives = tmp_0[0];
  rtb_R_clean_0 = tmp_0[2];
  for (i = 0; i <= 0; i += 2) {
    tmp_b = _mm_loadu_pd(&A[i + 3]);
    tmp_a = _mm_loadu_pd(&A[i]);
    tmp_7 = _mm_loadu_pd(&A[i + 6]);
    tmp_8 = _mm_loadu_pd(&rtb_MatrixMultiply7[i]);
    _mm_storeu_pd(&rtb_MatrixMultiply7_0[i], _mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_b, _mm_set1_pd(b_s)), _mm_mul_pd(tmp_a, _mm_set1_pd
      (Desired_Cable_Forces_Derivatives))), _mm_mul_pd(tmp_7, _mm_set1_pd
      (rtb_R_clean_0))), tmp_8));
    _mm_storeu_pd(&s[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    rtb_MatrixMultiply7_0[i] = ((A[i + 3] * b_s + A[i] *
      Desired_Cable_Forces_Derivatives) + A[i + 6] * rtb_R_clean_0) +
      rtb_MatrixMultiply7[i];
  }

  rtb_Sum_f_tmp_0 = 0.0;
  rtb_Sum_f_tmp_1 = 0.0;
  rtb_Sum_f_tmp_2 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Sum_f_tmp[3 * i]),
      _mm_set1_pd(rtb_Sum_p[i])), _mm_set_pd(rtb_Sum_f_tmp_1, rtb_Sum_f_tmp_0));
    _mm_storeu_pd(&tmp_9[0], tmp_b);
    rtb_Sum_f_tmp_0 = tmp_9[0];
    rtb_Sum_f_tmp_1 = tmp_9[1];
    rtb_Sum_f_tmp_2 += rtb_Sum_f_tmp[3 * i + 2] * rtb_Sum_p[i];
  }

  b_s = sqrt((Base_Model_U.Desired_Cable_Forces[9] *
              Base_Model_U.Desired_Cable_Forces[9] +
              Base_Model_U.Desired_Cable_Forces[10] *
              Base_Model_U.Desired_Cable_Forces[10]) +
             Base_Model_U.Desired_Cable_Forces[11] *
             Base_Model_U.Desired_Cable_Forces[11]);
  if (b_s > Base_Model_P.Saturation3_UpperSat) {
    b_s = Base_Model_P.Saturation3_UpperSat;
  } else if (b_s < Base_Model_P.Saturation3_LowerSat) {
    b_s = Base_Model_P.Saturation3_LowerSat;
  }

  tmp_b = _mm_set1_pd(b_s);
  _mm_storeu_pd(&rtb_Force_Sum[0], _mm_div_pd(_mm_loadu_pd
    (&Base_Model_U.Desired_Cable_Forces[9]), tmp_b));
  rtb_Force_Sum[2] = Base_Model_U.Desired_Cable_Forces[11] / b_s;
  for (i = 0; i < 3; i++) {
    tmp_a = _mm_sub_pd(_mm_loadu_pd(&Base_Model_P.Constant3_Value[3 * i]),
                       _mm_mul_pd(_mm_loadu_pd(&rtb_Force_Sum[0]), _mm_set1_pd
      (rtb_Force_Sum[i])));
    _mm_storeu_pd(&tmp[3 * i], tmp_a);
    jj = 3 * i + 2;
    tmp[jj] = Base_Model_P.Constant3_Value[jj] - rtb_Force_Sum[2] *
      rtb_Force_Sum[i];
  }

  Desired_Cable_Forces_Derivatives =
    Base_Model_U.Desired_Cable_Forces_Derivatives[10];
  rtb_R_clean_0 = Base_Model_U.Desired_Cable_Forces_Derivatives[9];
  rtb_MatrixMultiply5_idx_0 = Base_Model_U.Desired_Cable_Forces_Derivatives[11];
  for (i = 0; i <= 0; i += 2) {
    tmp_a = _mm_loadu_pd(&tmp[i + 3]);
    tmp_7 = _mm_loadu_pd(&tmp[i]);
    tmp_8 = _mm_loadu_pd(&tmp[i + 6]);
    _mm_storeu_pd(&s[i], _mm_div_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(tmp_a,
      _mm_set1_pd(Desired_Cable_Forces_Derivatives)), _mm_mul_pd(tmp_7,
      _mm_set1_pd(rtb_R_clean_0))), _mm_mul_pd(tmp_8, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_0))), tmp_b));
    _mm_storeu_pd(&rtb_MatrixMultiply7[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    s[i] = ((tmp[i + 3] * Desired_Cable_Forces_Derivatives + tmp[i] *
             rtb_R_clean_0) + tmp[i + 6] * rtb_MatrixMultiply5_idx_0) / b_s;
  }

  rtb_MatrixMultiply5_idx_0 = 0.0;
  rtb_MatrixMultiply5_idx_1 = 0.0;
  rtb_MatrixMultiply5_idx_2 = 0.0;
  for (i = 0; i < 3; i++) {
    b_s = Attachment_Point_Vectors[i + 9];
    tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Transpose[3 * i]),
      _mm_set1_pd(b_s)), _mm_set_pd(rtb_MatrixMultiply5_idx_1,
      rtb_MatrixMultiply5_idx_0));
    _mm_storeu_pd(&tmp_9[0], tmp_b);
    rtb_MatrixMultiply5_idx_0 = tmp_9[0];
    rtb_MatrixMultiply5_idx_1 = tmp_9[1];
    rtb_MatrixMultiply5_idx_2 += rtb_Transpose[3 * i + 2] * b_s;
  }

  smax = Attachment_Point_Vectors[9];
  Attachment_Point_Vectors_0 = Attachment_Point_Vectors[10];
  Attachment_Point_Vectors_1 = Attachment_Point_Vectors[11];
  for (i = 0; i <= 0; i += 2) {
    tmp_b = _mm_loadu_pd(&rtb_R_clean[i]);
    tmp_a = _mm_loadu_pd(&rtb_R_clean[i + 3]);
    tmp_7 = _mm_loadu_pd(&rtb_R_clean[i + 6]);
    tmp_8 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]);
    tmp_6 = _mm_loadu_pd(&s[i]);
    tmp_5 = _mm_set1_pd(Cable_Resting_Length);
    tmp_8 = _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(tmp_7, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_2)), _mm_add_pd(_mm_mul_pd(tmp_a, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_1)), _mm_mul_pd(tmp_b, _mm_set1_pd
      (rtb_MatrixMultiply5_idx_0)))), tmp_8), _mm_mul_pd(tmp_5, tmp_6));
    _mm_storeu_pd(&rtb_MatrixMultiply7[i], tmp_8);
    tmp_6 = _mm_loadu_pd(&rtb_Force_Sum[i]);
    tmp_3 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]);
    tmp_4 = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[i]);
    _mm_storeu_pd(&tmp_0[i], _mm_sub_pd(_mm_add_pd(_mm_add_pd(_mm_add_pd
      (_mm_mul_pd(tmp_7, _mm_set1_pd(Attachment_Point_Vectors_1)), _mm_add_pd
       (_mm_mul_pd(tmp_a, _mm_set1_pd(Attachment_Point_Vectors_0)), _mm_mul_pd
        (tmp_b, _mm_set1_pd(smax)))), _mm_mul_pd(tmp_6, tmp_5)), tmp_3), tmp_4));
    tmp_b = _mm_loadu_pd(&Base_Model_DW.UnitDelay_DSTATE_c[i]);
    _mm_storeu_pd(&rtb_Sum_p[i], _mm_sub_pd(tmp_8, tmp_b));
  }

  for (i = 2; i < 3; i++) {
    rtb_R_clean_0 = rtb_R_clean[i];
    b_s = rtb_R_clean_0 * rtb_MatrixMultiply5_idx_0;
    Desired_Cable_Forces_Derivatives = rtb_R_clean_0 * smax;
    rtb_R_clean_0 = rtb_R_clean[i + 3];
    b_s += rtb_R_clean_0 * rtb_MatrixMultiply5_idx_1;
    Desired_Cable_Forces_Derivatives += rtb_R_clean_0 *
      Attachment_Point_Vectors_0;
    rtb_R_clean_0 = rtb_R_clean[i + 6];
    b_s = ((rtb_R_clean_0 * rtb_MatrixMultiply5_idx_2 + b_s) +
           Base_Model_DW.DiscreteTimeIntegrator_DSTATE[i]) +
      Cable_Resting_Length * s[i];
    rtb_MatrixMultiply7[i] = b_s;
    tmp_0[i] = (((rtb_R_clean_0 * Attachment_Point_Vectors_1 +
                  Desired_Cable_Forces_Derivatives) + rtb_Force_Sum[i] *
                 Cable_Resting_Length) +
                Base_Model_DW.DiscreteTimeIntegrator1_DSTATE[i]) -
      Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[i];
    rtb_Sum_p[i] = b_s - Base_Model_DW.UnitDelay_DSTATE_c[i];
  }

  b_s = tmp_0[1];
  Desired_Cable_Forces_Derivatives = tmp_0[0];
  rtb_R_clean_0 = tmp_0[2];
  for (i = 0; i <= 0; i += 2) {
    tmp_b = _mm_loadu_pd(&A[i + 3]);
    tmp_a = _mm_loadu_pd(&A[i]);
    tmp_7 = _mm_loadu_pd(&A[i + 6]);
    tmp_8 = _mm_loadu_pd(&rtb_MatrixMultiply7[i]);
    _mm_storeu_pd(&rtb_Force_Sum[i], _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd
      (tmp_b, _mm_set1_pd(b_s)), _mm_mul_pd(tmp_a, _mm_set1_pd
      (Desired_Cable_Forces_Derivatives))), _mm_mul_pd(tmp_7, _mm_set1_pd
      (rtb_R_clean_0))), tmp_8));
    _mm_storeu_pd(&s[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    rtb_Force_Sum[i] = ((A[i + 3] * b_s + A[i] *
                         Desired_Cable_Forces_Derivatives) + A[i + 6] *
                        rtb_R_clean_0) + rtb_MatrixMultiply7[i];
  }

  b_s = 0.0;
  Desired_Cable_Forces_Derivatives = 0.0;
  rtb_R_clean_0 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Sum_f_tmp[3 * i]),
      _mm_set1_pd(rtb_Sum_p[i])), _mm_set_pd(Desired_Cable_Forces_Derivatives,
      b_s));
    _mm_storeu_pd(&tmp_9[0], tmp_b);
    b_s = tmp_9[0];
    Desired_Cable_Forces_Derivatives = tmp_9[1];
    rtb_R_clean_0 += rtb_Sum_f_tmp[3 * i + 2] * rtb_Sum_p[i];
  }

  s[2] = rtb_R_clean_0;
  s[1] = Desired_Cable_Forces_Derivatives;
  s[0] = b_s;
  tmp[0] = Base_Model_P.Constant_Value_m4;
  tmp[3] = Base_Model_P.Gain_Gain_i * Attachment_Point_Vectors[2];
  tmp[6] = Attachment_Point_Vectors[1];
  tmp[1] = Attachment_Point_Vectors[2];
  tmp[4] = Base_Model_P.Constant_Value_m4;
  _mm_storeu_pd(&tmp_9[0], _mm_mul_pd(_mm_set_pd(Base_Model_P.Gain2_Gain_e,
    Base_Model_P.Gain1_Gain_dl), _mm_loadu_pd(&Attachment_Point_Vectors[0])));
  tmp[7] = tmp_9[0];
  tmp[2] = tmp_9[1];
  tmp[5] = Attachment_Point_Vectors[0];
  tmp[8] = Base_Model_P.Constant_Value_m4;
  rtb_Sum_f_tmp[0] = Base_Model_P.Constant_Value_f;
  rtb_Sum_f_tmp[3] = Base_Model_P.Gain_Gain_g * Attachment_Point_Vectors[5];
  rtb_Sum_f_tmp[6] = Attachment_Point_Vectors[4];
  rtb_Sum_f_tmp[1] = Attachment_Point_Vectors[5];
  rtb_Sum_f_tmp[4] = Base_Model_P.Constant_Value_f;
  _mm_storeu_pd(&tmp_9[0], _mm_mul_pd(_mm_set_pd(Base_Model_P.Gain2_Gain_a,
    Base_Model_P.Gain1_Gain_g), _mm_loadu_pd(&Attachment_Point_Vectors[3])));
  rtb_Sum_f_tmp[7] = tmp_9[0];
  rtb_Sum_f_tmp[2] = tmp_9[1];
  rtb_Sum_f_tmp[5] = Attachment_Point_Vectors[3];
  rtb_Sum_f_tmp[8] = Base_Model_P.Constant_Value_f;
  for (i = 0; i < 3; i++) {
    rtb_Sum_p[i] = rtb_Force_Sum[i] + s[i];
    rtb_Transpose[3 * i] = rtb_R_clean[i];
    b_ix = 3 * i + 1;
    rtb_Transpose[b_ix] = rtb_R_clean[i + 3];
    ix = 3 * i + 2;
    rtb_Transpose[ix] = rtb_R_clean[i + 6];
    b_s = 0.0;
    Desired_Cable_Forces_Derivatives = 0.0;
    rtb_R_clean_0 = 0.0;
    for (jj = 0; jj < 3; jj++) {
      iy = 3 * i + jj;
      smax = rtb_Transpose[iy];
      tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&tmp[3 * jj]), _mm_set1_pd(smax)),
                         _mm_set_pd(Desired_Cable_Forces_Derivatives, b_s));
      _mm_storeu_pd(&tmp_9[0], tmp_b);
      b_s = tmp_9[0];
      Desired_Cable_Forces_Derivatives = tmp_9[1];
      rtb_R_clean_0 += tmp[3 * jj + 2] * smax;
      tmp_1[iy] = 0.0;
    }

    A[ix] = rtb_R_clean_0;
    A[b_ix] = Desired_Cable_Forces_Derivatives;
    A[3 * i] = b_s;
    b_s = tmp_1[3 * i];
    Desired_Cable_Forces_Derivatives = tmp_1[b_ix];
    rtb_R_clean_0 = tmp_1[ix];
    for (jj = 0; jj < 3; jj++) {
      smax = rtb_Transpose[3 * i + jj];
      tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Sum_f_tmp[3 * jj]),
        _mm_set1_pd(smax)), _mm_set_pd(Desired_Cable_Forces_Derivatives, b_s));
      _mm_storeu_pd(&tmp_9[0], tmp_b);
      b_s = tmp_9[0];
      Desired_Cable_Forces_Derivatives = tmp_9[1];
      rtb_R_clean_0 += rtb_Sum_f_tmp[3 * jj + 2] * smax;
    }

    tmp_1[ix] = rtb_R_clean_0;
    tmp_1[b_ix] = Desired_Cable_Forces_Derivatives;
    tmp_1[3 * i] = b_s;
  }

  b_s = 0.0;
  Desired_Cable_Forces_Derivatives = 0.0;
  rtb_R_clean_0 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&A[3 * i]), _mm_set1_pd
      (Base_Model_U.Desired_Cable_Forces[i])), _mm_set_pd
                       (Desired_Cable_Forces_Derivatives, b_s));
    _mm_storeu_pd(&tmp_9[0], tmp_b);
    b_s = tmp_9[0];
    Desired_Cable_Forces_Derivatives = tmp_9[1];
    rtb_R_clean_0 += A[3 * i + 2] * Base_Model_U.Desired_Cable_Forces[i];
    tmp_0[i] = 0.0;
  }

  s[2] = rtb_R_clean_0;
  s[1] = Desired_Cable_Forces_Derivatives;
  s[0] = b_s;
  tmp[0] = Base_Model_P.Constant_Value_i;
  tmp[3] = Base_Model_P.Gain_Gain_c * Attachment_Point_Vectors[8];
  tmp[6] = Attachment_Point_Vectors[7];
  tmp[1] = Attachment_Point_Vectors[8];
  tmp[4] = Base_Model_P.Constant_Value_i;
  _mm_storeu_pd(&tmp_9[0], _mm_mul_pd(_mm_set_pd(Base_Model_P.Gain2_Gain_m,
    Base_Model_P.Gain1_Gain_f), _mm_loadu_pd(&Attachment_Point_Vectors[6])));
  tmp[7] = tmp_9[0];
  tmp[2] = tmp_9[1];
  tmp[5] = Attachment_Point_Vectors[6];
  tmp[8] = Base_Model_P.Constant_Value_i;
  rtb_Sum_f_tmp[0] = Base_Model_P.Constant_Value_g4;
  rtb_Sum_f_tmp[3] = Base_Model_P.Gain_Gain_ii * Attachment_Point_Vectors[11];
  rtb_Sum_f_tmp[6] = Attachment_Point_Vectors[10];
  rtb_Sum_f_tmp[1] = Attachment_Point_Vectors[11];
  rtb_Sum_f_tmp[4] = Base_Model_P.Constant_Value_g4;
  _mm_storeu_pd(&tmp_9[0], _mm_mul_pd(_mm_set_pd(Base_Model_P.Gain2_Gain_oh,
    Base_Model_P.Gain1_Gain_j), _mm_loadu_pd(&Attachment_Point_Vectors[9])));
  rtb_Sum_f_tmp[7] = tmp_9[0];
  rtb_Sum_f_tmp[2] = tmp_9[1];
  rtb_Sum_f_tmp[5] = Attachment_Point_Vectors[9];
  rtb_Sum_f_tmp[8] = Base_Model_P.Constant_Value_g4;
  for (i = 0; i < 3; i++) {
    b_s = Base_Model_U.Desired_Cable_Forces[i + 3];
    tmp_0[0] += tmp_1[3 * i] * b_s;
    jj = 3 * i + 1;
    tmp_0[1] += tmp_1[jj] * b_s;
    iy = 3 * i + 2;
    tmp_0[2] += tmp_1[iy] * b_s;
    b_s = 0.0;
    Desired_Cable_Forces_Derivatives = 0.0;
    rtb_R_clean_0 = 0.0;
    for (b_ix = 0; b_ix < 3; b_ix++) {
      ix = 3 * i + b_ix;
      smax = rtb_Transpose[ix];
      tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&tmp[3 * b_ix]), _mm_set1_pd
        (smax)), _mm_set_pd(Desired_Cable_Forces_Derivatives, b_s));
      _mm_storeu_pd(&tmp_9[0], tmp_b);
      b_s = tmp_9[0];
      Desired_Cable_Forces_Derivatives = tmp_9[1];
      rtb_R_clean_0 += tmp[3 * b_ix + 2] * smax;
      rtb_R_clean[ix] = 0.0;
    }

    A[iy] = rtb_R_clean_0;
    A[jj] = Desired_Cable_Forces_Derivatives;
    A[3 * i] = b_s;
    b_s = rtb_R_clean[3 * i];
    Desired_Cable_Forces_Derivatives = rtb_R_clean[jj];
    rtb_R_clean_0 = rtb_R_clean[iy];
    for (b_ix = 0; b_ix < 3; b_ix++) {
      smax = rtb_Transpose[3 * i + b_ix];
      tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_Sum_f_tmp[3 * b_ix]),
        _mm_set1_pd(smax)), _mm_set_pd(Desired_Cable_Forces_Derivatives, b_s));
      _mm_storeu_pd(&tmp_9[0], tmp_b);
      b_s = tmp_9[0];
      Desired_Cable_Forces_Derivatives = tmp_9[1];
      rtb_R_clean_0 += rtb_Sum_f_tmp[3 * b_ix + 2] * smax;
    }

    rtb_R_clean[iy] = rtb_R_clean_0;
    rtb_R_clean[jj] = Desired_Cable_Forces_Derivatives;
    rtb_R_clean[3 * i] = b_s;
  }

  b_s = Base_Model_U.Desired_Cable_Forces[7];
  Desired_Cable_Forces_Derivatives = Base_Model_U.Desired_Cable_Forces[6];
  rtb_R_clean_0 = Base_Model_U.Desired_Cable_Forces[8];
  for (i = 0; i <= 0; i += 2) {
    tmp_b = _mm_loadu_pd(&A[i + 3]);
    tmp_a = _mm_loadu_pd(&A[i]);
    tmp_7 = _mm_loadu_pd(&A[i + 6]);
    tmp_8 = _mm_loadu_pd(&s[i]);
    tmp_6 = _mm_loadu_pd(&tmp_0[i]);
    _mm_storeu_pd(&tmp_2[i], _mm_add_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(tmp_b,
      _mm_set1_pd(b_s)), _mm_mul_pd(tmp_a, _mm_set1_pd
      (Desired_Cable_Forces_Derivatives))), _mm_mul_pd(tmp_7, _mm_set1_pd
      (rtb_R_clean_0))), _mm_add_pd(tmp_8, tmp_6)));
    _mm_storeu_pd(&rtb_Force_Sum[i], _mm_set1_pd(0.0));
  }

  for (i = 2; i < 3; i++) {
    tmp_2[i] = ((A[i + 3] * b_s + A[i] * Desired_Cable_Forces_Derivatives) + A[i
                + 6] * rtb_R_clean_0) + (s[i] + tmp_0[i]);
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
  b_s = 0.0;
  Desired_Cable_Forces_Derivatives = 0.0;
  rtb_R_clean_0 = 0.0;
  for (i = 0; i < 3; i++) {
    smax = Base_Model_U.Desired_Cable_Forces[i + 9];
    tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&rtb_R_clean[3 * i]), _mm_set1_pd
      (smax)), _mm_set_pd(Desired_Cable_Forces_Derivatives, b_s));
    _mm_storeu_pd(&tmp_9[0], tmp_b);
    b_s = tmp_9[0];
    Desired_Cable_Forces_Derivatives = tmp_9[1];
    _mm_storeu_pd(&tmp_9[0], _mm_add_pd(_mm_mul_pd(_mm_set_pd
      (Load_Inertia_Matrix[3 * i], rtb_R_clean[3 * i + 2]), _mm_set_pd
      (Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[i], smax)), _mm_set_pd
      (rtb_MatrixMultiply5_idx_0, rtb_R_clean_0)));
    rtb_R_clean_0 = tmp_9[0];
    rtb_MatrixMultiply5_idx_0 = tmp_9[1];
    _mm_storeu_pd(&tmp_9[0], _mm_add_pd(_mm_mul_pd(_mm_loadu_pd
      (&Load_Inertia_Matrix[3 * i + 1]), _mm_set1_pd
      (Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[i])), _mm_set_pd
      (rtb_MatrixMultiply5_idx_2, rtb_MatrixMultiply5_idx_1)));
    rtb_MatrixMultiply5_idx_1 = tmp_9[0];
    rtb_MatrixMultiply5_idx_2 = tmp_9[1];
  }

  smax = -g * Load_Mass;
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
       Base_Model_U.Desired_Cable_Forces[9]) + smax *
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
       Base_Model_U.Desired_Cable_Forces[10]) + smax *
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
       Base_Model_U.Desired_Cable_Forces[11]) + smax *
      Base_Model_P.Constant_Value_fy[2]) -
     Base_Model_DW.DiscreteTimeIntegrator_DSTATE[2] * Load_Damping_Coef) /
    Load_Mass * Base_Model_P.DiscreteTimeIntegrator_gainval;
  for (i = 0; i <= 6; i += 2) {
    tmp_b = _mm_loadu_pd(&U[i]);
    tmp_a = _mm_loadu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i]);
    _mm_storeu_pd(&Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i], _mm_add_pd
                  (_mm_mul_pd(_mm_set1_pd
      (Base_Model_P.DiscreteTimeIntegrator2_gainval), tmp_b), tmp_a));
  }

  for (i = 8; i < 9; i++) {
    Base_Model_DW.DiscreteTimeIntegrator2_DSTATE[i] +=
      Base_Model_P.DiscreteTimeIntegrator2_gainval * U[i];
  }

  rt_invd3x3_snf(Load_Inertia_Matrix, tmp);
  rtb_Force_Sum[0] = ((rtb_MatrixMultiply7[1] * rtb_MatrixMultiply5_idx_2 -
                       rtb_MatrixMultiply5_idx_1 * rtb_MatrixMultiply7[2]) +
                      (tmp_2[0] + b_s)) -
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0] * Load_Damping_Coef;
  b_s = 0.0;
  rtb_Force_Sum[1] = ((rtb_MatrixMultiply5_idx_0 * rtb_MatrixMultiply7[2] -
                       rtb_MatrixMultiply7[0] * rtb_MatrixMultiply5_idx_2) +
                      (tmp_2[1] + Desired_Cable_Forces_Derivatives)) -
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[1] * Load_Damping_Coef;
  Desired_Cable_Forces_Derivatives = 0.0;
  rtb_Force_Sum[2] = ((rtb_MatrixMultiply7[0] * rtb_MatrixMultiply5_idx_1 -
                       rtb_MatrixMultiply5_idx_0 * rtb_MatrixMultiply7[1]) +
                      (tmp_2[2] + rtb_R_clean_0)) -
    Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2] * Load_Damping_Coef;
  rtb_R_clean_0 = 0.0;
  for (i = 0; i < 3; i++) {
    tmp_b = _mm_add_pd(_mm_mul_pd(_mm_loadu_pd(&tmp[3 * i]), _mm_set1_pd
      (rtb_Force_Sum[i])), _mm_set_pd(Desired_Cable_Forces_Derivatives, b_s));
    _mm_storeu_pd(&tmp_9[0], tmp_b);
    b_s = tmp_9[0];
    Desired_Cable_Forces_Derivatives = tmp_9[1];
    rtb_R_clean_0 += tmp[3 * i + 2] * rtb_Force_Sum[i];
  }

  tmp_b = _mm_set_pd(Base_Model_P.DiscreteTimeIntegrator1_gainval_g,
                     Base_Model_P.DiscreteTimeIntegrator3_gainval);
  _mm_storeu_pd(&tmp_9[0], _mm_add_pd(_mm_mul_pd(tmp_b, _mm_set_pd
    (Base_Model_DW.UnitDelay_DSTATE[0], b_s)), _mm_set_pd
    (Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[0],
     Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0])));
  Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[0] = tmp_9[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[0] = tmp_9[1];
  tmp_a = _mm_set_pd(Base_Model_P.DiscreteTimeIntegrator1_gainval_i,
                     Base_Model_P.DiscreteTimeIntegrator1_gainval_n);
  _mm_storeu_pd(&tmp_9[0], _mm_add_pd(_mm_mul_pd(tmp_a, _mm_set_pd
    (Base_Model_DW.UnitDelay_DSTATE_k[0], Base_Model_DW.UnitDelay_DSTATE_b[0])),
    _mm_set_pd(Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[0],
               Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[0])));
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[0] = tmp_9[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[0] = tmp_9[1];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[0] +=
    Base_Model_P.DiscreteTimeIntegrator1_gainval_g0 *
    Base_Model_DW.UnitDelay_DSTATE_c[0];
  _mm_storeu_pd(&tmp_9[0], _mm_add_pd(_mm_set_pd(rtb_Force_Sum_0[0],
    rtb_Transpose1[0]), _mm_set_pd(rtb_Sum_f_tmp_3, rtb_R_clean_1)));
  Base_Model_DW.UnitDelay_DSTATE[0] = tmp_9[0];
  Base_Model_DW.UnitDelay_DSTATE_b[0] = tmp_9[1];
  Base_Model_DW.UnitDelay_DSTATE_k[0] = rtb_MatrixMultiply7_0[0] +
    rtb_Sum_f_tmp_0;
  Base_Model_DW.UnitDelay_DSTATE_c[0] = rtb_Sum_p[0];
  _mm_storeu_pd(&tmp_9[0], _mm_add_pd(_mm_mul_pd(tmp_b, _mm_set_pd
    (Base_Model_DW.UnitDelay_DSTATE[1], Desired_Cable_Forces_Derivatives)),
    _mm_set_pd(Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[1],
               Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[1])));
  Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[1] = tmp_9[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[1] = tmp_9[1];
  _mm_storeu_pd(&tmp_9[0], _mm_add_pd(_mm_mul_pd(tmp_a, _mm_set_pd
    (Base_Model_DW.UnitDelay_DSTATE_k[1], Base_Model_DW.UnitDelay_DSTATE_b[1])),
    _mm_set_pd(Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[1],
               Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[1])));
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[1] = tmp_9[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[1] = tmp_9[1];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[1] +=
    Base_Model_P.DiscreteTimeIntegrator1_gainval_g0 *
    Base_Model_DW.UnitDelay_DSTATE_c[1];
  _mm_storeu_pd(&tmp_9[0], _mm_add_pd(_mm_set_pd(rtb_Force_Sum_0[1],
    rtb_Transpose1[1]), _mm_set_pd(rtb_Sum_f_tmp_4, rtb_R_clean_2)));
  Base_Model_DW.UnitDelay_DSTATE[1] = tmp_9[0];
  Base_Model_DW.UnitDelay_DSTATE_b[1] = tmp_9[1];
  Base_Model_DW.UnitDelay_DSTATE_k[1] = rtb_MatrixMultiply7_0[1] +
    rtb_Sum_f_tmp_1;
  Base_Model_DW.UnitDelay_DSTATE_c[1] = rtb_Sum_p[1];
  _mm_storeu_pd(&tmp_9[0], _mm_add_pd(_mm_mul_pd(tmp_b, _mm_set_pd
    (Base_Model_DW.UnitDelay_DSTATE[2], rtb_R_clean_0)), _mm_set_pd
    (Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[2],
     Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2])));
  Base_Model_DW.DiscreteTimeIntegrator3_DSTATE[2] = tmp_9[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_f[2] = tmp_9[1];
  _mm_storeu_pd(&tmp_9[0], _mm_add_pd(_mm_mul_pd(tmp_a, _mm_set_pd
    (Base_Model_DW.UnitDelay_DSTATE_k[2], Base_Model_DW.UnitDelay_DSTATE_b[2])),
    _mm_set_pd(Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[2],
               Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[2])));
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_g[2] = tmp_9[0];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_l[2] = tmp_9[1];
  Base_Model_DW.DiscreteTimeIntegrator1_DSTATE_fr[2] +=
    Base_Model_P.DiscreteTimeIntegrator1_gainval_g0 *
    Base_Model_DW.UnitDelay_DSTATE_c[2];
  _mm_storeu_pd(&tmp_9[0], _mm_add_pd(_mm_set_pd(rtb_Force_Sum_0[2],
    rtb_Transpose1[2]), _mm_set_pd(rtb_Sum_f_tmp_5, U_0)));
  Base_Model_DW.UnitDelay_DSTATE[2] = tmp_9[0];
  Base_Model_DW.UnitDelay_DSTATE_b[2] = tmp_9[1];
  Base_Model_DW.UnitDelay_DSTATE_k[2] = rtb_MatrixMultiply7_0[2] +
    rtb_Sum_f_tmp_2;
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
