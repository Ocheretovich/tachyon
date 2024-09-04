// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/field_merkle_tree.h"

#include "gtest/gtest.h"

#include "tachyon/crypto/hashes/sponge/padding_free_sponge.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_external_matrix.h"
#include "tachyon/crypto/hashes/sponge/truncated_permutation.h"
#include "tachyon/math/finite_fields/baby_bear/poseidon2.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::crypto {

constexpr size_t kRate = 8;
constexpr size_t kChunk = 8;
constexpr size_t kN = 2;

using F = math::BabyBear;
using PackedF = math::PackedBabyBear;
using Poseidon2 =
    Poseidon2Sponge<Poseidon2ExternalMatrix<Poseidon2Plonky3ExternalMatrix<F>>>;
using PackedPoseidon2 = Poseidon2Sponge<
    Poseidon2ExternalMatrix<Poseidon2Plonky3ExternalMatrix<PackedF>>>;
using MyHasher = PaddingFreeSponge<Poseidon2, kRate, kChunk>;
using MyPackedHasher = PaddingFreeSponge<PackedPoseidon2, kRate, kChunk>;
using MyCompressor = TruncatedPermutation<Poseidon2, kChunk, kN>;
using MyPackedCompressor = TruncatedPermutation<PackedPoseidon2, kChunk, kN>;
using Tree = FieldMerkleTree<F, kChunk>;

namespace {

class FieldMerkleTreeTest : public math::FiniteFieldTest<PackedF> {
 public:
  void SetUp() override {
    Poseidon2Config<F> config = Poseidon2Config<F>::CreateCustom(
        15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
    Poseidon2 sponge(std::move(config));
    hasher_ = MyHasher(sponge);
    compressor_ = MyCompressor(std::move(sponge));

    Poseidon2Config<PackedF> packed_config =
        Poseidon2Config<PackedF>::CreateCustom(
            15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
    PackedPoseidon2 packed_sponge(std::move(packed_config));
    packed_hasher_ = MyPackedHasher(packed_sponge);
    packed_compressor_ = MyPackedCompressor(std::move(packed_sponge));
  }

 protected:
  MyHasher hasher_;
  MyCompressor compressor_;
  MyPackedHasher packed_hasher_;
  MyPackedCompressor packed_compressor_;
};

}  // namespace

TEST_F(FieldMerkleTreeTest, CommitSingle1x8) {
  math::RowMajorMatrix<F> matrix{
      {F(2)}, {F(1)}, {F(2)}, {F(2)}, {F(0)}, {F(0)}, {F(1)}, {F(0)},
  };
  std::vector<math::RowMajorMatrix<F>> matrices = {matrix};

  Tree tree = Tree::Build(hasher_, packed_hasher_, compressor_,
                          packed_compressor_, std::move(matrices));

  auto h0_1 = compressor_.Compress(std::vector<std::array<F, kChunk>>{
      hasher_.Hash(std::vector<F>{matrix(0, 0)}),
      hasher_.Hash(std::vector<F>{matrix(1, 0)})});
  for (const auto& digest : h0_1) {
    LOG(ERROR) << digest;
  }
  auto h2_3 = compressor_.Compress(std::vector<std::array<F, kChunk>>{
      hasher_.Hash(std::vector<F>{matrix(2, 0)}),
      hasher_.Hash(std::vector<F>{matrix(3, 0)})});
  for (const auto& digest : h2_3) {
    LOG(ERROR) << digest;
  }
  auto h4_5 = compressor_.Compress(std::vector<std::array<F, kChunk>>{
      hasher_.Hash(std::vector<F>{matrix(4, 0)}),
      hasher_.Hash(std::vector<F>{matrix(5, 0)})});
  for (const auto& digest : h4_5) {
    LOG(ERROR) << digest;
  }
  auto h6_7 = compressor_.Compress(std::vector<std::array<F, kChunk>>{
      hasher_.Hash(std::vector<F>{matrix(6, 0)}),
      hasher_.Hash(std::vector<F>{matrix(7, 0)})});
  for (const auto& digest : h6_7) {
    LOG(ERROR) << digest;
  }
  auto h0_3 =
      compressor_.Compress(std::vector<std::array<F, kChunk>>{h0_1, h2_3});
  for (const auto& digest : h0_3) {
    LOG(ERROR) << digest;
  }
  auto h4_7 =
      compressor_.Compress(std::vector<std::array<F, kChunk>>{h4_5, h6_7});
  for (const auto& digest : h4_7) {
    LOG(ERROR) << digest;
  }
  auto expected =
      compressor_.Compress(std::vector<std::array<F, kChunk>>{h0_3, h4_7});
  for (const auto& digest : expected) {
    LOG(ERROR) << digest;
  }
  EXPECT_EQ(tree.GetRoot(), expected);
}

TEST_F(FieldMerkleTreeTest, CommitSingle2x2) {
  math::RowMajorMatrix<F> matrix{
      {F(0), F(1)},
      {F(2), F(1)},
  };
  std::vector<math::RowMajorMatrix<F>> matrices = {matrix};

  Tree tree = Tree::Build(hasher_, packed_hasher_, compressor_,
                          packed_compressor_, std::move(matrices));

  auto expected = compressor_.Compress(std::vector<std::array<F, kChunk>>{
      hasher_.Hash(std::vector<F>{matrix(0, 0), matrix(0, 1)}),
      hasher_.Hash(std::vector<F>{matrix(1, 0), matrix(1, 1)})});
  EXPECT_EQ(tree.GetRoot(), expected);
}

TEST_F(FieldMerkleTreeTest, CommitSingle2x3) {
  math::RowMajorMatrix<F> matrix{
      {F(0), F(1)},
      {F(2), F(1)},
      {F(2), F(2)},
  };
  std::vector<math::RowMajorMatrix<F>> matrices = {matrix};

  Tree tree = Tree::Build(hasher_, packed_hasher_, compressor_,
                          packed_compressor_, std::move(matrices));
  std::array<F, kChunk> default_digest =
      base::CreateArray<kChunk>([]() { return F::Zero(); });
  auto h0_3 = compressor_.Compress(std::vector<std::array<F, kChunk>>{
      hasher_.Hash(std::vector<F>{matrix(0, 0), matrix(0, 1)}),
      hasher_.Hash(std::vector<F>{matrix(1, 0), matrix(1, 1)})});
  auto h4_5 = compressor_.Compress(std::vector<std::array<F, kChunk>>{
      hasher_.Hash(std::vector<F>{matrix(2, 0), matrix(2, 1)}),
      default_digest});
  auto expected =
      compressor_.Compress(std::vector<std::array<F, kChunk>>{h0_3, h4_5});
  EXPECT_EQ(tree.GetRoot(), expected);
}

TEST_F(FieldMerkleTreeTest, CommitMixed) {
  math::RowMajorMatrix<F> matrix{
      {F(0), F(1)},
      {F(2), F(1)},
      {F(2), F(2)},
  };
  math::RowMajorMatrix<F> matrix2{
      {F(1), F(2), F(1)},
      {F(0), F(2), F(2)},
  };
  std::vector<math::RowMajorMatrix<F>> matrices = {matrix, matrix2};

  Tree tree = Tree::Build(hasher_, packed_hasher_, compressor_,
                          packed_compressor_, std::move(matrices));
  std::array<F, kChunk> default_digest =
      base::CreateArray<kChunk>([]() { return F::Zero(); });
  auto h0_3 = compressor_.Compress(std::vector<std::array<F, kChunk>>{
      hasher_.Hash(std::vector<F>{matrix(0, 0), matrix(0, 1)}),
      hasher_.Hash(std::vector<F>{matrix(1, 0), matrix(1, 1)})});
  auto h4_6 =
      hasher_.Hash(std::vector<F>{matrix2(0, 0), matrix2(0, 1), matrix2(0, 2)});
  auto h7_8 = compressor_.Compress(std::vector<std::array<F, kChunk>>{
      hasher_.Hash(std::vector<F>{matrix(2, 0), matrix(2, 1)}),
      default_digest});
  auto h9_11 =
      hasher_.Hash(std::vector<F>{matrix2(1, 0), matrix2(1, 1), matrix2(1, 2)});
  auto h0_6 =
      compressor_.Compress(std::vector<std::array<F, kChunk>>{h0_3, h4_6});
  auto h7_11 =
      compressor_.Compress(std::vector<std::array<F, kChunk>>{h7_8, h9_11});
  auto expected =
      compressor_.Compress(std::vector<std::array<F, kChunk>>{h0_6, h7_11});
  EXPECT_EQ(tree.GetRoot(), expected);
}

TEST_F(FieldMerkleTreeTest, CommitEitherOrder) {
  math::RowMajorMatrix<F> matrix = math::RowMajorMatrix<F>::Random(3, 2);
  math::RowMajorMatrix<F> matrix2 = math::RowMajorMatrix<F>::Random(2, 3);

  std::vector<math::RowMajorMatrix<F>> matrices = {matrix, matrix2};
  Tree tree = Tree::Build(hasher_, packed_hasher_, compressor_,
                          packed_compressor_, std::move(matrices));

  std::vector<math::RowMajorMatrix<F>> matrices2 = {matrix, matrix2};
  Tree tree2 = Tree::Build(hasher_, packed_hasher_, compressor_,
                           packed_compressor_, std::move(matrices2));
  EXPECT_EQ(tree.GetRoot(), tree2.GetRoot());
}

}  // namespace tachyon::crypto
