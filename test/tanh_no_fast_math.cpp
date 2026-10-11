// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// EIGEN_FAST_MATH=0 selects the accurate float tanh kernel, which backends such as the generic clang one vectorize
// whatever the setting.
#define EIGEN_FAST_MATH 0

#include "main.h"

#include <cstring>

namespace {

numext::int64_t float_ulp_distance(float a, float b) {
  numext::int32_t ia, ib;
  std::memcpy(&ia, &a, sizeof(float));
  std::memcpy(&ib, &b, sizeof(float));
  // Map the sign-magnitude bit patterns onto a monotonic integer line.
  const numext::int64_t la = ia < 0 ? numext::int64_t(INT32_MIN) - ia : ia;
  const numext::int64_t lb = ib < 0 ? numext::int64_t(INT32_MIN) - ib : ib;
  return la > lb ? la - lb : lb - la;
}

template <typename Packet>
void tanh_float_accuracy() {
  constexpr int PacketSize = internal::unpacket_traits<Packet>::size;
  // Every third float in [0.5, 10], where the fast kernel is up to 6 ulps off; the accurate one is within 2.
  const numext::uint32_t first = 0x3f000000u, last = 0x41200000u;
  float x[PacketSize], y[PacketSize];
  numext::int64_t max_error = 0;
  float worst = 0;
  for (numext::uint32_t bits = first; bits < last;) {
    for (int k = 0; k < PacketSize; ++k, bits += 3) {
      std::memcpy(&x[k], &bits, sizeof(float));
      if (k % 2 == 1) x[k] = -x[k];
    }
    internal::pstoreu(y, internal::ptanh_float(internal::ploadu<Packet>(x)));
    for (int k = 0; k < PacketSize; ++k) {
      const numext::int64_t error = float_ulp_distance(y[k], static_cast<float>(std::tanh(double(x[k]))));
      if (error > max_error) {
        max_error = error;
        worst = x[k];
      }
    }
  }
  if (max_error > 2) std::cout << "tanh(" << worst << ") is " << max_error << " ulps off" << std::endl;
  VERIFY(max_error <= 2);

  const float inf = NumTraits<float>::infinity();
  const float special[] = {0.0f, -0.0f, inf, -inf, NumTraits<float>::quiet_NaN(), 20.0f, -20.0f, 1e-30f};
  for (const float s : special) {
    for (int k = 0; k < PacketSize; ++k) x[k] = s;
    internal::pstoreu(y, internal::ptanh_float(internal::ploadu<Packet>(x)));
    const float expected = std::tanh(s);
    const bool success = ((numext::isnan)(expected) && (numext::isnan)(y[0])) ||
                         (y[0] == expected && std::signbit(y[0]) == std::signbit(expected));
    if (!success) std::cout << "tanh(" << s << ") = " << y[0] << " != " << expected << std::endl;
    VERIFY(success);
  }
}

}  // namespace

EIGEN_DECLARE_TEST(tanh_no_fast_math) {
  CALL_SUBTEST(tanh_float_accuracy<internal::packet_traits<float>::type>());
  CALL_SUBTEST(tanh_float_accuracy<float>());
}
