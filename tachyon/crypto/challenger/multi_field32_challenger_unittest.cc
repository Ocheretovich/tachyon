// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/crypto/challenger/multi_field32_challenger.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/crypto/hashes/sponge/poseidon2/param_traits/poseidon2_baby_bear.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/param_traits/poseidon2_bn254.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_params.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_external_matrix.h"

namespace tachyon::crypto {

using Params = Poseidon2Params<math::bn254::Fr, 2, 5>;
using Poseidon2 = Poseidon2Sponge<
    Poseidon2ExternalMatrix<Poseidon2Plonky3ExternalMatrix<math::bn254::Fr>>,
    Params>;

namespace {

class MultiField32ChallengerTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    math::BabyBear::Init();
    math::bn254::Fr::Init();
  }

  void SetUp() override {
    auto config = Poseidon2Config<Params>::Create(
        crypto::GetPoseidon2InternalDiagonalArray<Params>());
    Poseidon2 sponge(std::move(config));
    challenger_.reset(new MultiField32Challenger<math::BabyBear, Poseidon2>(
        std::move(sponge)));
  }

 protected:
  std::unique_ptr<MultiField32Challenger<math::BabyBear, Poseidon2>>
      challenger_;
};

}  // namespace

TEST_F(MultiField32ChallengerTest, Sample) {
  for (uint32_t i = 0; i < 20; ++i) {
    challenger_->Observe(math::BabyBear(i));
  }

  math::BabyBear answers[] = {
      math::BabyBear(72199253),   math::BabyBear(733473132),
      math::BabyBear(442816494),  math::BabyBear(326641700),
      math::BabyBear(1342573676), math::BabyBear(1242755868),
      math::BabyBear(887300172),  math::BabyBear(1831922292),
      math::BabyBear(1518709680),
  };
  for (size_t i = 0; i < std::size(answers); ++i) {
    EXPECT_EQ(challenger_->Sample(), answers[i]);
  }
}

TEST_F(MultiField32ChallengerTest, Grind) {
  const uint32_t kBits = 3;
  math::BabyBear witness =
      challenger_->Grind(kBits, base::Range<uint32_t>(0, 100));
  EXPECT_TRUE(challenger_->CheckWitness(kBits, witness));
}

}  // namespace tachyon::crypto
