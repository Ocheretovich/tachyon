#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs_baby_bear.h"

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/numeric/bits.h"
#include "third_party/icicle/src/merkle-tree/merkle.cu.cc"  // NOLINT(build/include)
#include "third_party/icicle/src/merkle-tree/mmcs.cu.cc"  // NOLINT(build/include)

#include "tachyon/base/bits.h"
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_logging.h"

cudaError_t tachyon_babybear_mmcs_commit_cuda(
    const ::matrix::Matrix<::babybear::scalar_t>* leaves,
    unsigned int number_of_inputs, ::babybear::scalar_t* digests,
    const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>* hasher,
    const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>*
        compression,
    const ::merkle_tree::TreeBuilderConfig& tree_config) {
  return ::merkle_tree::mmcs_commit<::babybear::scalar_t, ::babybear::scalar_t>(
      leaves, number_of_inputs, digests, *hasher, *compression, tree_config);
}

namespace tachyon::crypto {

template <>
bool IcicleMMCS<math::BabyBear>::DoCommit(
    std::vector<math::RowMajorMatrix<math::BabyBear>>&& matrices,
    std::vector<std::vector<std::vector<math::BabyBear>>>&& outputs,
    void* icicle_poseidon2) {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif
  size_t max_tree_height = 0;
  size_t number_of_leaves = 0;
  for (const auto& matrix : matrices) {
    size_t tree_height = base::bits::Log2Ceiling(
        absl::bit_ceil(static_cast<size_t>(matrix.rows())));
    max_tree_height = std::max(max_tree_height, tree_height);
    number_of_leaves += matrix.size();
  }

  auto allocate_and_convert =
      [](const math::RowMajorMatrix<math::BabyBear>& matrix,
         std::vector<::babybear::scalar_t>& dest_data)
      -> ::babybear::scalar_t* {
    absl::Span<const math::BabyBear> matrix_span(matrix.data(), matrix.size());
    auto icicle_matrix_span =
        reinterpret_cast<const ::babybear::scalar_t*>(std::data(matrix_span));

    for (size_t i = 0; i < matrix_span.size(); ++i) {
      dest_data[i] =
          ::babybear::scalar_t::from_montgomery(icicle_matrix_span[i]);
    }

    ::babybear::scalar_t* d_matrix;
    cudaMalloc(&d_matrix, matrix.size() * sizeof(::babybear::scalar_t));
    cudaMemcpy(d_matrix, dest_data.data(),
               matrix.size() * sizeof(::babybear::scalar_t),
               cudaMemcpyHostToDevice);

    return d_matrix;
  };

  std::unique_ptr<::matrix::Matrix<::babybear::scalar_t>[]> leaves(
      new ::matrix::Matrix<::babybear::scalar_t>[matrices.size()]);

  size_t idx = 0;
  for (const auto& matrix : matrices) {
    std::vector<::babybear::scalar_t> dest_data(matrix.size());
    ::babybear::scalar_t* d_matrix = allocate_and_convert(matrix, dest_data);

    leaves[idx] = {
        d_matrix,
        static_cast<size_t>(matrix.cols()),
        static_cast<size_t>(matrix.rows()),
    };
    LOG(ERROR) << "matrix.size(): " << matrix.size();
    LOG(ERROR) << "matrix.cols(): " << matrix.cols();
    LOG(ERROR) << "matrix.rows(): " << matrix.rows();
    ++idx;
  }

  config_->keep_rows = max_tree_height + 1;
  config_->digest_elements = 8;
  size_t digests_len = ::merkle_tree::get_digests_len(
      config_->keep_rows - 1, config_->arity, config_->digest_elements);

  std::unique_ptr<::babybear::scalar_t[]> icicle_digest(
      new ::babybear::scalar_t[digests_len]);
  LOG(ERROR) << "number_of_leaves: " << number_of_leaves;
  cudaError_t error = tachyon_babybear_mmcs_commit_cuda(
      leaves.get(), matrices.size(), icicle_digest.get(),
      reinterpret_cast<::poseidon2::Poseidon2<::babybear::scalar_t>*>(
          icicle_poseidon2),
      reinterpret_cast<::poseidon2::Poseidon2<::babybear::scalar_t>*>(
          icicle_poseidon2),
      *config_);
  for (size_t idx = 0; idx < digests_len; ++idx) {
    LOG(ERROR) << "init icicle_digest[" << idx << "]: " << icicle_digest[idx];
  }

  outputs.reserve(config_->keep_rows);
  size_t previous_number_of_element = 0;
  for (size_t layer_idx = 0; layer_idx <= max_tree_height; ++layer_idx) {
    std::vector<std::vector<math::BabyBear>> digest_layer;
    size_t number_of_node = 1 << (max_tree_height - layer_idx);
    digest_layer.reserve(number_of_node);

    for (size_t node_idx = 0; node_idx < number_of_node; ++node_idx) {
      std::vector<math::BabyBear> digest;
      digest.reserve(config_->digest_elements);

      for (size_t element_idx = 0; element_idx < config_->digest_elements;
           ++element_idx) {
        size_t idx = previous_number_of_element +
                     config_->digest_elements * node_idx + element_idx;
        icicle_digest[idx] =
            ::babybear::scalar_t::to_montgomery(icicle_digest[idx]);
        LOG(ERROR) << "icicle_digest[" << idx << "]: " << icicle_digest[idx];
        digest.emplace_back(
            *reinterpret_cast<math::BabyBear*>(&icicle_digest[idx]));
      }
      digest_layer.emplace_back(std::move(digest));
    }
    outputs.emplace_back(std::move(digest_layer));
    previous_number_of_element += number_of_node * config_->digest_elements;
  }

  return error == cudaSuccess;
}

}  // namespace tachyon::crypto
