// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/Dense>
#include <contrib/Eigen/MatrixFunctions>

struct MissingShape {};
struct PartialShape {};
template <int Rows, int Cols>
struct ShapeProbe;
namespace Eigen {
namespace internal {
template <int Rows, int Cols>
struct traits<ShapeProbe<Rows, Cols>> : traits<Matrix<double, Rows, Cols>> {};
template <>
struct traits<PartialShape> {
  static constexpr int RowsAtCompileTime = 3;
};
}  // namespace internal
}  // namespace Eigen

template <int Rows, int Cols>
struct ShapeProbe : MatrixBase<ShapeProbe<Rows, Cols>> {
  using Nested = const ShapeProbe&;
  mutable int row_reads = 0;
  mutable int col_reads = 0;
  Index rows() const {
    ++row_reads;
    return 2;
  }
  Index cols() const {
    ++col_reads;
    return 2;
  }
};

EIGEN_DECLARE_TEST(shape_checks) {
  using namespace Eigen::internal;
  STATIC_CHECK((dimension_relation<3, 3>::Proven));
  STATIC_CHECK((!dimension_relation<3, 2>::Compatible));
  STATIC_CHECK((dimension_relation<Dynamic, 3>::Compatible));
  STATIC_CHECK((!dimension_relation<Dynamic, 3>::Known));
  STATIC_CHECK((compile_time_shape<MissingShape>::Rows == Dynamic));
  STATIC_CHECK((compile_time_shape<PartialShape>::Rows == 3));
  STATIC_CHECK((compile_time_shape<PartialShape>::Cols == Dynamic));
  STATIC_CHECK((compile_time_shape<Matrix<double, Dynamic, 1, 0, 3, 1>>::Size == Dynamic));
  STATIC_CHECK((matching_sizes<Matrix<double, 2, 3>, Matrix<double, 3, 2>>::Proven));
  STATIC_CHECK((square_dimensions<const Matrix3d&>::Proven));
  int reads = 0;
  struct RuntimeDimensions {
    mutable int accesses = 0;
    Index rows() const {
      ++accesses;
      return 2;
    }
    Index cols() const {
      ++accesses;
      return 3;
    }
  } runtime_a, runtime_b;
  EIGEN_ASSERT_SAME_ROWS(runtime_a, runtime_b);
  EIGEN_ASSERT_SAME_COLS(runtime_a, runtime_b);
#ifndef EIGEN_NO_DEBUG
  VERIFY_IS_EQUAL(runtime_a.accesses, 2);
  VERIFY_IS_EQUAL(runtime_b.accesses, 2);
#endif
  EIGEN_SHAPE_ASSERT((dimension_relation<3, 3>::Proven), ++reads == 1);
  VERIFY_IS_EQUAL(reads, 0);
#ifndef EIGEN_NO_DEBUG
  EIGEN_SHAPE_ASSERT((dimension_relation<Dynamic, 3>::Proven), ++reads == 1);
  VERIFY_IS_EQUAL(reads, 1);
  VERIFY_RAISES_ASSERT([] { EIGEN_SHAPE_ASSERT(false, false); }());
#endif
}
