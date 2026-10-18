// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "packetmath_test_shared.h"

void check_trig_reduction() {
  // MPFR 4.2.1, 256-bit precision, rounded to binary64. Include both sides of the medium/large reduction cutoffs.
  const double cases[][4] = {
      {0.29999999999999999, 0.29552020666133955, 0.95533648912560598, 0.30933624960962325},
      {4.7123889803846897, -1, -1.8369701987210297e-16, 5443746451065123},
      {14.999999999999998, 0.65028784015711827, -0.75968791285882009, -0.85599340090852183},
      {15, 0.65028784015711683, -0.75968791285882131, -0.85599340090851872},
      {15.000000000000002, 0.65028784015711549, -0.75968791285882242, -0.85599340090851572},
      {20, 0.91294525072762767, 0.40808206181339196, 2.2371609442247422},
      {100, -0.50636564110975879, 0.86231887228768389, -0.58721391515692911},
      {123.456, -0.80393736857282394, -0.5947139710921574, 1.3518050821917635},
      {1000, 0.82687954053200252, 0.56237907629070294, 1.4703241557027185},
      {12345.678900000001, -0.70344192126325633, 0.71075274421521484, -0.98971397154430851},
      {123456.789, -0.99866408234322457, 0.051672532718701383, -19.326787943215852},
      {1000000, -0.34999350217129294, 0.93675212753314474, -0.37362445398759903},
      {10000000, 0.4205477931907825, -0.90727038618173961, -0.46353082785018906},
      {100000000, 0.93163902710972601, -0.36338508935569053, -2.5637789067283774},
      {1000000000, 0.54584344944869956, 0.83788718136390239, 0.65145220214514132},
      {10000000000, -0.48750602508751067, 0.87311962267685606, -0.55834963781124181},
      {100000000000, 0.92869366049659197, 0.37084779216471114, 2.5042448144982212},
      {1000000000000, -0.61123870237688949, 0.79144630185289022, -0.77230596813187613},
      {6907362400000, 0.5754525274479354, 0.81783518428457402, 0.70362896889955862},
      {10000000000000, -0.28888529481752512, 0.95736371690083999, -0.30175082856983471},
      {99999999999999.984, -0.19410480015836853, -0.9809807982603328, 0.1978680933435121},
      {100000000000000, -0.20940830749645231, -0.97782828796853249, 0.21415652428250395},
      {100000000000000.02, -0.22466069079959752, -0.97443705492425092, 0.23055433869667633},
      {1000000000000000, 0.85827279317023586, -0.51319373778697031, -1.672414782127583},
  };
  const Index count = sizeof(cases) / sizeof(cases[0]);
  ArrayXd x(2 * count), sin_ref(2 * count), cos_ref(2 * count), tan_ref(2 * count);
  for (Index i = 0; i < count; ++i) {
    x[2 * i] = cases[i][0];
    x[2 * i + 1] = -cases[i][0];
    sin_ref[2 * i] = cases[i][1];
    sin_ref[2 * i + 1] = -cases[i][1];
    cos_ref[2 * i] = cos_ref[2 * i + 1] = cases[i][2];
    tan_ref[2 * i] = cases[i][3];
    tan_ref[2 * i + 1] = -cases[i][3];
  }
  const ArrayXd sine = x.sin(), cosine = x.cos(), tangent = x.tan();
  VERIFY(test::areWithinUlps(sin_ref.data(), sine.data(), static_cast<int>(x.size()), 4));
  VERIFY(test::areWithinUlps(cos_ref.data(), cosine.data(), static_cast<int>(x.size()), 4));
  VERIFY(test::areWithinUlps(tan_ref.data(), tangent.data(), static_cast<int>(x.size()), 4));
}

template <typename Packet>
void check_error_free_transformations() {
  const double low = std::ldexp(1.0, -54);
  Packet hi, lo;
  internal::twosum(internal::pset1<Packet>(low), internal::pset1<Packet>(1.0), hi, lo);
  VERIFY_IS_EQUAL(internal::pfirst(hi), 1.0);
  VERIFY_IS_EQUAL(internal::pfirst(lo), low);
  internal::fast_twosum(internal::pset1<Packet>(1.0), internal::pset1<Packet>(low), hi, lo);
  VERIFY_IS_EQUAL(internal::pfirst(hi), 1.0);
  VERIFY_IS_EQUAL(internal::pfirst(lo), low);
  const Packet x = internal::pset1<Packet>(1.0 + std::ldexp(1.0, -30));
  internal::twoprod(x, x, hi, lo);
  VERIFY_IS_EQUAL(internal::pfirst(hi), 1.0 + std::ldexp(1.0, -29));
  VERIFY_IS_EQUAL(internal::pfirst(lo), std::ldexp(1.0, -60));
  VERIFY_IS_EQUAL(internal::pfirst(internal::twoprod_low(x, x, hi)), std::ldexp(1.0, -60));
}

EIGEN_DECLARE_TEST(trig_reduction_fastmath) {
  CALL_SUBTEST(check_trig_reduction());
  CALL_SUBTEST(check_error_free_transformations<double>());
  CALL_SUBTEST(check_error_free_transformations<internal::packet_traits<double>::type>());
}
