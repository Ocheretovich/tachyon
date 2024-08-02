// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#include "tachyon/zk/plonk/vanishing/vanishing_utils.h"

#include <vector>

#include "tachyon/base/memory/reusing_allocator.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::zk::plonk {

namespace {

using F = math::bn254::Fr;

class VanishingUtilsTest : public math::FiniteFieldTest<F> {
 public:
  constexpr static size_t N = size_t{1} << 4;
  constexpr static size_t kMaxDegree = N - 1;

  using Domain = math::UnivariateEvaluationDomain<F, kMaxDegree>;
  using Poly = typename Domain::DensePoly;
  using Coeffs = typename Poly::Coefficients;
  using Evals = typename Domain::Evals;
};

}  // namespace

TEST_F(VanishingUtilsTest, GetZeta) {
  F zeta = GetZeta<F>();
  EXPECT_EQ(zeta.Pow(3), F::One());
  F halo2_zeta = GetHalo2Zeta<F>();
  EXPECT_EQ(halo2_zeta.Pow(3), F::One());
}

TEST_F(VanishingUtilsTest, BuildExtendedColumnWithColumns) {
  std::vector<std::vector<F>> columns =
      base::CreateVector(4, [](size_t i) { return std::vector<F>(N, F(i)); });

  std::vector<F, base::memory::ReusingAllocator<F>> extended =
      BuildExtendedColumnWithColumns(columns);
  EXPECT_EQ(extended.size(), 4 * N);
  for (size_t i = 0; i < extended.size(); ++i) {
    EXPECT_EQ(F(i % 4), extended[i]);
  }
}

}  // namespace tachyon::zk::plonk
