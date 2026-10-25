// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2019 Joel Holdsworth <joel.holdsworth@vcatechnology.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include <sstream>

#include "main.h"

template <typename Scalar>
struct check_ostream_impl {
  static void run() {
    const Array<Scalar, 1, 1> array(123);
    std::ostringstream ss;
    ss << array;
    VERIFY(ss.str() == "123");

    check_ostream_impl<std::complex<Scalar> >::run();
  };
};

template <>
struct check_ostream_impl<bool> {
  static void run() {
    const Array<bool, 1, 2> array(1, 0);
    std::ostringstream ss;
    ss << array;
    VERIFY(ss.str() == "1  0");
  };
};

template <typename Scalar>
struct check_ostream_impl<std::complex<Scalar> > {
  static void run() {
    const Array<std::complex<Scalar>, 1, 1> array(std::complex<Scalar>(12, 34));
    std::ostringstream ss;
    ss << array;
    VERIFY(ss.str() == "(12,34)");
  };
};

template <typename Scalar>
static void check_ostream() {
  check_ostream_impl<Scalar>::run();
}

static std::string check_ioformat(const std::string& rowSeparator = "\n", const std::string& rowPrefix = "[",
                                  const std::string& rowSuffix = "]") {
  const auto matrix = (Matrix2i() << 1, 20, 300, 4000).finished();
  std::ostringstream ss;
  const auto flags{0};
  const auto coeffSeparator{", "};
  const auto matPrefix{"ABC"};
  const auto matSuffix{"XYZ"};
  const auto fill{'*'};
  const auto fmt =
      IOFormat(StreamPrecision, flags, coeffSeparator, rowSeparator, rowPrefix, rowSuffix, matPrefix, matSuffix, fill);
  ss << matrix.format(fmt);
  return ss.str();
}

static void check_ioformat_standard() {
  const auto found{check_ioformat()};
  const auto expected =
      "ABC[***1, **20]\n"
      "   [*300, 4000]XYZ";
  VERIFY_IS_EQUAL(found, expected);
}

static void check_ioformat_row_separator_newline() {
  const auto found{check_ioformat("\nNext\n")};
  const auto expected =
      "ABC[***1, **20]\n"
      "Next\n"
      "   [*300, 4000]XYZ";
  VERIFY_IS_EQUAL(found, expected);
}

static void check_ioformat_row_prefix_newline() {
  const auto found{check_ioformat("\n", "[\n", "]")};
  const auto expected =
      "ABC[\n"
      "***1, **20]\n"
      "   [\n"
      "*300, 4000]XYZ";
  VERIFY_IS_EQUAL(found, expected);
}

static void check_ioformat_row_suffix_newline() {
  const auto found{check_ioformat("\n", "[", "\n]")};
  const auto expected =
      "ABC[***1, **20\n"
      "]\n"
      "   [*300, 4000\n"
      "]XYZ";
  VERIFY_IS_EQUAL(found, expected);
}

EIGEN_DECLARE_TEST(io) {
  CALL_SUBTEST(check_ostream<bool>());
  CALL_SUBTEST(check_ostream<float>());
  CALL_SUBTEST(check_ostream<double>());
  CALL_SUBTEST(check_ostream<Eigen::numext::int8_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::uint8_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::int16_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::uint16_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::int32_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::uint32_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::int64_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::uint64_t>());
  CALL_SUBTEST(check_ioformat_standard());
  CALL_SUBTEST(check_ioformat_row_separator_newline());
  CALL_SUBTEST(check_ioformat_row_prefix_newline());
  CALL_SUBTEST(check_ioformat_row_suffix_newline());
}
