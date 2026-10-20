// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_MATH_FUNCTIONS_NEON_H
#define EIGEN_MATH_FUNCTIONS_NEON_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(Packet2f)
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(Packet4f)

#if EIGEN_ARCH_ARM64 && EIGEN_HAS_ARM64_FP16
template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet4hf ptanh<Packet4hf>(const Packet4hf& x) {
  // Convert to float, call the float ptanh, and then convert back.
  return vcvt_f16_f32(ptanh<Packet4f>(vcvt_f32_f16(x)));
}

template <>
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Packet8hf ptanh<Packet8hf>(const Packet8hf& x) {
  // Convert each 4 half types to float, call the float ptanh, and then convert back.
  return vcombine_f16(vcvt_f16_f32(ptanh<Packet4f>(vcvt_f32_f16(vget_low_f16(x)))),
                      vcvt_f16_f32(ptanh<Packet4f>(vcvt_f32_f16(vget_high_f16(x)))));
}
#endif  // EIGEN_ARCH_ARM64 && EIGEN_HAS_ARM64_FP16

// psqrt is specialized below instead of taking the float round trip.
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, pcos)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, psin)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, psinh)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, pcosh)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, pasinh)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, pacosh)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, pexp)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, pexp2)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, pexpm1)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, plog)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, plog1p)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, plog2)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, plog10)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, preciprocal)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, prsqrt)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, pcbrt)
BF16_PACKET_FUNCTION(Packet4f, Packet4bf, ptanh)

template <>
EIGEN_STRONG_INLINE Packet4bf psqrt<Packet4bf>(const Packet4bf& a) {
  // y = x r with r = FRSQRTE(x), then one FRSQRTS step y <- y (3 - y r) / 2. FRSQRTE is specified
  // bit for bit, and rounding y to bfloat16 equals bfloat16(sqrtf(x)) for all 65536 inputs
  // (checked exhaustively); the raw estimate alone misrounds 7366 of them. The narrowing adds half
  // a bfloat16 ulp and truncates. That is round half up, which differs from F32ToBf16 only on an
  // exact float midpoint, and no input produces one; NaNs stay NaN since their low half is zero.
  // The estimate route gives NaN for 0 and inf, so those lanes keep the input.
  Packet4f x = Bf16ToF32(a);
  Packet4f r = vrsqrteq_f32(x);
  Packet4f y = pmul(x, r);
  y = pmul(y, vrsqrtsq_f32(y, r));
  Packet4bf ybf = vshrn_n_u32(vaddq_u32(vreinterpretq_u32_f32(y), vdupq_n_u32(0x8000)), 16);
#if EIGEN_ARCH_ARM64
  uint16x4_t zero = vceq_u16(vshl_n_u16(a, 1), vdup_n_u16(0));
#else
  // ARMv7 NEON flushes subnormals, so the estimate of one is inf. Keep them as the input, which
  // is what the float compare in psqrt_float_common did for them.
  uint16x4_t zero = vceq_u16(vand_u16(a, vdup_n_u16(0x7f80)), vdup_n_u16(0));
#endif
  uint16x4_t pass = vorr_u16(zero, vceq_u16(a, vdup_n_u16(0x7f80)));
  return vbsl_u16(pass, a, ybf);
}

template <>
EIGEN_STRONG_INLINE Packet4bf pfrexp(const Packet4bf& a, Packet4bf& exponent) {
  // Both results are exact: the mantissa keeps the input's significand, and the exponent is a
  // small integer.
  Packet4f fexponent;
  const Packet4bf out = F32ToBf16Truncate(pfrexp<Packet4f>(Bf16ToF32(a), fexponent));
  exponent = F32ToBf16Truncate(fexponent);
  return out;
}

template <>
EIGEN_STRONG_INLINE Packet4bf pldexp(const Packet4bf& a, const Packet4bf& exponent) {
  return F32ToBf16(pldexp<Packet4f>(Bf16ToF32(a), Bf16ToF32(exponent)));
}

//---------- double ----------

#if EIGEN_ARCH_ARM64

EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_DOUBLE(Packet2d)

#endif

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_MATH_FUNCTIONS_NEON_H
