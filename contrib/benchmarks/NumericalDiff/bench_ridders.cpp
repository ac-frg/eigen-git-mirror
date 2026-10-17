// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <contrib/Eigen/NumericalDiff>

template <typename Scalar_, int Options>
struct CubicFunctor {
  using Scalar = Scalar_;
  using InputType = Eigen::Matrix<Scalar, 2, 1>;
  using ValueType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using JacobianType = Eigen::Matrix<Scalar, Eigen::Dynamic, 2, Options>;
  static constexpr int InputsAtCompileTime = 2;
  static constexpr int ValuesAtCompileTime = Eigen::Dynamic;
  Eigen::Index count;

  Eigen::Index values() const { return count; }
  int operator()(const InputType& x, ValueType& y) const {
    for (Eigen::Index i = 0; i < count; ++i) y[i] = Scalar(i + 1) * x[0] * x[0] * x[0] + x[1] * x[1] * x[1];
    return 0;
  }
};

template <typename Scalar, int Options>
static void BM_Ridders(benchmark::State& state) {
  using Functor = CubicFunctor<Scalar, Options>;
  const Eigen::Index count = state.range(0);
  Eigen::Ridders<Functor> derivative(Functor{count});
  typename Eigen::Ridders<Functor>::Parameters parameters;
  parameters.maxSteps = static_cast<int>(state.range(1));
  derivative.setParameters(parameters);
  typename Functor::InputType x;
  x << Scalar(0.5), Scalar(-1);
  typename Functor::JacobianType jac(count, 2);
  derivative.df(x, jac);
  for (Eigen::Index i = 0; i < count; ++i) {
    const Scalar expected = Scalar(i + 1) * Scalar(0.75);
    // Central differences divide rounded function values by a step of order 0.01.
    const Scalar bound = Scalar(512) * Eigen::NumTraits<Scalar>::epsilon() * Eigen::numext::maxi(expected, Scalar(3));
    if (!(Eigen::numext::abs(jac(i, 0) - expected) <= bound && Eigen::numext::abs(jac(i, 1) - Scalar(3)) <= bound)) {
      state.SkipWithError("Incorrect cubic Jacobian");
      return;
    }
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(derivative.df(x, jac));
    benchmark::DoNotOptimize(jac.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK_TEMPLATE(BM_Ridders, float, Eigen::ColMajor)->ArgsProduct({{1, 16, 128, 1024}, {2, 10}});
BENCHMARK_TEMPLATE(BM_Ridders, double, Eigen::ColMajor)->ArgsProduct({{1, 16, 128, 1024}, {2, 10}});
BENCHMARK_TEMPLATE(BM_Ridders, double, Eigen::RowMajor)->ArgsProduct({{1, 16, 128, 1024}, {2, 10}});
