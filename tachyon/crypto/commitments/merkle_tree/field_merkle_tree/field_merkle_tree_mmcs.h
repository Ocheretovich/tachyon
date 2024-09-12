// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_FIELD_MERKLE_TREE_MMCS_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_FIELD_MERKLE_TREE_MMCS_H_

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/numeric/bits.h"

#include "tachyon/base/bits.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/profiler.h"
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/field_merkle_tree.h"
#include "tachyon/crypto/commitments/mixed_matrix_commitment_scheme.h"
#include "tachyon/math/finite_fields/extension_field_traits_forward.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"

#if TACHYON_CUDA
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/icicle/icicle_poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_config.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_params.h"
#include "tachyon/device/gpu/gpu_memory.h"
#include "tachyon/device/gpu/scoped_mem_pool.h"
#include "tachyon/device/gpu/scoped_stream.h"
#endif

namespace tachyon::crypto {

template <typename F, typename Hasher, typename PackedHasher,
          typename Compressor, typename PackedCompressor, size_t N>
class FieldMerkleTreeMMCS final
    : public MixedMatrixCommitmentScheme<FieldMerkleTreeMMCS<
          F, Hasher, PackedHasher, Compressor, PackedCompressor, N>> {
 public:
  using PrimeField =
      std::conditional_t<math::FiniteFieldTraits<F>::kIsExtensionField,
                         typename math::ExtensionFieldTraits<F>::BasePrimeField,
                         F>;
  using Commitment = std::array<PrimeField, N>;
  using Digest = Commitment;
  using ProverData = FieldMerkleTree<F, N>;
  using Proof = std::vector<Digest>;

  FieldMerkleTreeMMCS() = default;
  FieldMerkleTreeMMCS(const Hasher& hasher, const PackedHasher& packed_hasher,
                      const Compressor& compressor,
                      const PackedCompressor& packed_compressor)
      : hasher_(hasher),
        packed_hasher_(packed_hasher),
        compressor_(compressor),
        packed_compressor_(packed_compressor) {
#if TACHYON_CUDA
    SetupForGpu();
#endif
  }
  FieldMerkleTreeMMCS(Hasher&& hasher, PackedHasher&& packed_hasher,
                      Compressor&& compressor,
                      PackedCompressor&& packed_compressor)
      : hasher_(std::move(hasher)),
        packed_hasher_(std::move(packed_hasher)),
        compressor_(std::move(compressor)),
        packed_compressor_(std::move(packed_compressor)) {
#if TACHYON_CUDA
    SetupForGpu();
#endif
  }

#if TACHYON_CUDA
  void SetupForGpu() {
    if constexpr (IsIciclePoseidon2Supported<F> && IsIcicleMMCSSupported<F>) {
      if (poseidon2_gpu_) return;
      if (mmcs_gpu_) return;
      using Params = Poseidon2Params<F, 15, 7>;

      gpuMemPoolProps props = {gpuMemAllocationTypePinned,
                               gpuMemHandleTypeNone,
                               {gpuMemLocationTypeDevice, 0}};
      mem_pool_ = device::gpu::CreateMemPool(&props);

      uint64_t mem_pool_threshold = std::numeric_limits<uint64_t>::max();
      gpuError_t error = gpuMemPoolSetAttribute(
          mem_pool_.get(), gpuMemPoolAttrReleaseThreshold, &mem_pool_threshold);
      CHECK_EQ(error, gpuSuccess);
      stream_ = device::gpu::CreateStream();

      poseidon2_gpu_.reset(
          new IciclePoseidon2<F>(mem_pool_.get(), stream_.get()));

      auto config = Poseidon2Config<Params>::Create(
          GetPoseidon2InternalShiftArray<Params>());

      if (config.use_plonky3_internal_matrix) {
        math::Vector<F> internal_vector = math::Vector<F>(Params::kWidth);
        internal_vector[0] = F(F::Config::kModulus - 2);
        for (Eigen::Index i = 1; i < internal_vector.size(); ++i) {
          internal_vector[i] = F(uint32_t{1} << config.internal_shifts[i - 1]);
        }
        absl::Span<const F> internal_vector_span =
            absl::Span<const F>(internal_vector.data(), internal_vector.size());
        size_t capacity =
            Params::kFullRounds * Params::kWidth + Params::kPartialRounds;

        std::vector<F> ark_vector;
        ark_vector.reserve(capacity);
        Eigen::Index partial_rounds_start = Params::kFullRounds / 2;
        Eigen::Index partial_rounds_end =
            Params::kFullRounds / 2 + Params::kPartialRounds;
        for (Eigen::Index i = 0; i < config.ark.rows(); ++i) {
          if (i < partial_rounds_start || i >= partial_rounds_end) {
            for (Eigen::Index j = 0; j < config.ark.cols(); ++j) {
              ark_vector.push_back(config.ark(i, j));
            }
          } else {
            ark_vector.push_back(config.ark(i, 0));
          }
        }
        absl::Span<const F> ark_span =
            absl::Span<const F>(ark_vector.data(), ark_vector.size());
        if (poseidon2_gpu_->Create(Params::kWidth, Params::kRate,
                                   Params::kAlpha, Params::kPartialRounds,
                                   Params::kFullRounds, ark_span,
                                   internal_vector_span, Vendor::kPlonky3)) {
          mmcs_gpu_.reset(new IcicleMMCS<F>(mem_pool_.get(), stream_.get()));
          return;
        }
      } else {
        if (poseidon2_gpu_->Load(Params::kWidth, Params::kRate,
                                 Vendor::kHorizen)) {
          mmcs_gpu_.reset(new IcicleMMCS<F>(mem_pool_.get(), stream_.get()));
          return;
        }
      }

      LOG(ERROR) << "Failed poseidon2 gpu setup";
      poseidon2_gpu_.reset();
    }
  }
#endif

  const Hasher& hasher() const { return hasher_; }
  const PackedHasher& packed_hasher() const { return packed_hasher_; }
  const Compressor& compressor() const { return compressor_; }
  const PackedCompressor& packed_compressor() const {
    return packed_compressor_;
  }

 private:
  friend class MixedMatrixCommitmentScheme<FieldMerkleTreeMMCS<
      F, Hasher, PackedHasher, Compressor, PackedCompressor, N>>;

  struct IndexedDimensions {
    size_t index;
    math::Dimensions dimensions;

    // TODO(chokobole): This comparison is intentionally reversed to sort in
    // descending order, as powersort doesn't accept custom callbacks.
    bool operator<(const IndexedDimensions& other) const {
      return dimensions.height > other.dimensions.height;
    }
    bool operator<=(const IndexedDimensions& other) const {
      return dimensions.height >= other.dimensions.height;
    }
    bool operator>(const IndexedDimensions& other) const {
      return dimensions.height < other.dimensions.height;
    }

    std::string ToString() const {
      return absl::Substitute("($0, $1)", index, dimensions.ToString());
    }
  };

  [[nodiscard]] bool DoCommit(std::vector<math::RowMajorMatrix<F>>&& matrices,
                              Commitment* commitment,
                              ProverData* prover_data) const {
#if TACHYON_CUDA
    if constexpr (IsIciclePoseidon2Supported<F> && IsIcicleMMCSSupported<F>) {
      if (!poseidon2_gpu_ || !mmcs_gpu_) return false;

      std::vector<std::vector<std::vector<F>>> digest_layers_icicle;
      bool result = mmcs_gpu_->DoCommit(std::move(matrices),
                                        std::move(digest_layers_icicle),
                                        poseidon2_gpu_->data());

      std::vector<std::vector<Digest>> digest_layers;
      digest_layers.resize(digest_layers_icicle.size());
      for (size_t i = 0; i < digest_layers_icicle.size(); ++i) {
        digest_layers[i].resize(digest_layers_icicle[i].size());
        for (size_t j = 0; j < digest_layers_icicle[i].size(); ++j) {
          if (digest_layers_icicle[i][j].size() == N) {
            std::array<PrimeField, N> arr;

            std::move(digest_layers_icicle[i][j].begin(),
                      digest_layers_icicle[i][j].end(), arr.begin());

            digest_layers[i][j] = std::move(arr);
          }
        }
      }
      for (const auto& output : digest_layers_icicle) {
        for (const auto& layer : output) {
          for (const auto& element : layer) {
            LOG(ERROR) << "output: " << element;
          }
        }
      }

      if (result) {
        *prover_data =
            FieldMerkleTree(std::move(matrices), std::move(digest_layers));
        *commitment = prover_data->GetRoot();
        return true;
      }
    }
#endif
    TRACE_EVENT("ProofGeneration", "FieldMerkleTreeMMCS::DoCommit");
    *prover_data =
        FieldMerkleTree<F, N>::Build(hasher_, packed_hasher_, compressor_,
                                     packed_compressor_, std::move(matrices));
    *commitment = prover_data->GetRoot();

    return true;
  }

  const std::vector<math::RowMajorMatrix<F>>& DoGetMatrices(
      const ProverData& prover_data) const {
    return prover_data.leaves();
  }

  [[nodiscard]] bool DoCreateOpeningProof(size_t index,
                                          const ProverData& prover_data,
                                          std::vector<std::vector<F>>* openings,
                                          Proof* proof) const {
    TRACE_EVENT("ProofGeneration", "FieldMerkleTreeMMCS::DoCreateOpeningProof");
    size_t max_row_size = this->GetMaxRowSize(prover_data);
    uint32_t log_max_row_size = base::bits::Log2Ceiling(max_row_size);

    // TODO(chokobole): Is it able to be parallelized?
    *openings = base::Map(
        prover_data.leaves(),
        [log_max_row_size, index](const math::RowMajorMatrix<F>& matrix) {
          uint32_t log_row_size =
              base::bits::Log2Ceiling(static_cast<size_t>(matrix.rows()));
          uint32_t bits_reduced = log_max_row_size - log_row_size;
          size_t reduced_index = index >> bits_reduced;
          return base::CreateVector(matrix.cols(),
                                    [reduced_index, &matrix](size_t col) {
                                      return matrix(reduced_index, col);
                                    });
        });

    *proof =
        base::CreateVector(log_max_row_size, [&prover_data, index](size_t i) {
          // NOTE(chokobole): Let v be |index >> i|. If v is even, v ^ 1 is v
          // + 1. Otherwise, v ^ 1 is v - 1.
          return prover_data.digest_layers()[i][(index >> i) ^ 1];
        });

    return true;
  }

  [[nodiscard]] bool DoVerifyOpeningProof(
      const Commitment& commitment,
      absl::Span<const math::Dimensions> dimensions_list, size_t index,
      absl::Span<const std::vector<F>> opened_values,
      const Proof& proof) const {
    TRACE_EVENT("ProofVerification",
                "FieldMerkleTreeMMCS::DoVerifyOpeningProof");
    CHECK_EQ(dimensions_list.size(), opened_values.size());

    std::vector<IndexedDimensions> sorted_dimensions_list = base::Map(
        dimensions_list, [](size_t index, math::Dimensions dimensions) {
          return IndexedDimensions{index, dimensions};
        });

    base::StableSort(sorted_dimensions_list.begin(),
                     sorted_dimensions_list.end());

    absl::Span<const IndexedDimensions> remaining_dimensions_list =
        absl::MakeConstSpan(sorted_dimensions_list);

    size_t next_layer =
        absl::bit_ceil(remaining_dimensions_list.front().dimensions.height);
    size_t next_layer_size = CountLayers(next_layer, remaining_dimensions_list);
    Digest root = hasher_.Hash(GetOpenedValuesAsPrimeFieldVectors(
        opened_values, remaining_dimensions_list.subspan(0, next_layer_size)));
    remaining_dimensions_list.remove_prefix(next_layer_size);

    for (const Digest& sibling : proof) {
      Digest inputs[2];
      inputs[0] = (index & 1) == 0 ? root : sibling;
      inputs[1] = (index & 1) == 0 ? sibling : root;
      root = compressor_.Compress(inputs);

      index >>= 1;
      next_layer >>= 1;
      next_layer_size = CountLayers(next_layer, remaining_dimensions_list);
      if (next_layer_size > 0) {
        inputs[0] = std::move(root);
        inputs[1] = hasher_.Hash(GetOpenedValuesAsPrimeFieldVectors(
            opened_values,
            remaining_dimensions_list.subspan(0, next_layer_size)));
        remaining_dimensions_list.remove_prefix(next_layer_size);

        root = compressor_.Compress(inputs);
      }
    }
    LOG(ERROR) << "CUDA_TEST3";
    for (const auto& ele : root) {
      LOG(ERROR) << "root: " << ele;
    }

    for (const auto& ele : commitment) {
      LOG(ERROR) << "commitment: " << ele;
    }

    return root == commitment;
  }

  constexpr static size_t CountLayers(
      size_t target_height,
      absl::Span<const IndexedDimensions> dimensions_list) {
    TRACE_EVENT("Utils", "CountLayers");
    size_t ret = 0;
    for (size_t i = 0; i < dimensions_list.size(); ++i) {
      if (target_height == absl::bit_ceil(static_cast<size_t>(
                               dimensions_list[i].dimensions.height))) {
        ++ret;
      } else {
        break;
      }
    }
    return ret;
  }

  static std::vector<PrimeField> GetOpenedValuesAsPrimeFieldVectors(
      absl::Span<const std::vector<F>> opened_values,
      absl::Span<const IndexedDimensions> dimensions_list) {
    TRACE_EVENT("Utils", "GetOpenedValuesAsPrimeFieldVectors");
    if constexpr (math::FiniteFieldTraits<F>::kIsExtensionField) {
      static_assert(math::ExtensionFieldTraits<F>::kDegreeOverBasePrimeField ==
                    math::ExtensionFieldTraits<F>::kDegreeOverBaseField);
      size_t size = std::accumulate(
          dimensions_list.begin(), dimensions_list.end(), 0,
          [&opened_values](size_t acc, const IndexedDimensions& dimensions) {
            return acc + opened_values[dimensions.index].size();
          });
      std::vector<PrimeField> ret;
      ret.reserve(size *
                  math::ExtensionFieldTraits<F>::kDegreeOverBasePrimeField);
      for (size_t i = 0; i < dimensions_list.size(); ++i) {
        const std::vector<F>& elements =
            opened_values[dimensions_list[i].index];
        for (size_t j = 0; j < elements.size(); ++j) {
          const F& element = elements[j];
          for (size_t k = 0;
               k < math::ExtensionFieldTraits<F>::kDegreeOverBasePrimeField;
               ++k) {
            ret.push_back(element[k]);
          }
        }
      }
      return ret;
    } else {
      return base::FlatMap(
          dimensions_list,
          [&opened_values](const IndexedDimensions& dimensions) {
            return opened_values[dimensions.index];
          });
    }
  }

  Hasher hasher_;
  PackedHasher packed_hasher_;
  Compressor compressor_;
  PackedCompressor packed_compressor_;

#if TACHYON_CUDA
  device::gpu::ScopedMemPool mem_pool_;
  device::gpu::ScopedStream stream_;
  std::unique_ptr<IciclePoseidon2<F>> poseidon2_gpu_;
  std::unique_ptr<IcicleMMCS<F>> mmcs_gpu_;
#endif
};

template <typename F, typename Hasher, typename PackedHasher,
          typename Compressor, typename PackedCompressor, size_t N>
struct MixedMatrixCommitmentSchemeTraits<FieldMerkleTreeMMCS<
    F, Hasher, PackedHasher, Compressor, PackedCompressor, N>> {
 public:
  using Field = F;
  using PrimeField =
      std::conditional_t<math::FiniteFieldTraits<F>::kIsExtensionField,
                         typename math::ExtensionFieldTraits<F>::BasePrimeField,
                         F>;
  using Commitment = std::array<PrimeField, N>;
  using ProverData = FieldMerkleTree<F, N>;
  using Proof = std::vector<std::array<PrimeField, N>>;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_FIELD_MERKLE_TREE_MMCS_H_
