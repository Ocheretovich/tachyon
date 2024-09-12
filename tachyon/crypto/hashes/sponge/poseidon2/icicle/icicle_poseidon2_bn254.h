#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_BN254_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_BN254_H_

#include <stdint.h>

#include "third_party/icicle/include/curves/params/bn254.cu.h"
#include "third_party/icicle/include/poseidon2/poseidon2.cu.h"

extern "C" cudaError_t tachyon_bn254_poseidon2_create_cuda(
    ::poseidon2::Poseidon2<bn254::scalar_t>** poseidon, unsigned int width,
    unsigned int rate, unsigned int alpha, unsigned int internal_rounds,
    unsigned int external_rounds, const ::bn254::scalar_t* round_constants,
    const ::bn254::scalar_t* internal_matrix_diag,
    ::poseidon2::MdsType mds_type, ::poseidon2::DiffusionStrategy diffusion,
    ::device_context::DeviceContext& ctx);

extern "C" cudaError_t tachyon_bn254_poseidon2_load_cuda(
    ::poseidon2::Poseidon2<bn254::scalar_t>** poseidon, unsigned int width,
    unsigned int rate, ::poseidon2::MdsType mds_type,
    ::poseidon2::DiffusionStrategy diffusion,
    ::device_context::DeviceContext& ctx);

extern "C" cudaError_t tachyon_bn254_poseidon2_hash_many_cuda(
    const ::poseidon2::Poseidon2<::bn254::scalar_t>* poseidon,
    const ::bn254::scalar_t* inputs, ::bn254::scalar_t* output,
    unsigned int number_of_states, unsigned int input_block_len,
    unsigned int output_len, ::hash::HashConfig& cfg);

extern "C" cudaError_t tachyon_bn254_poseidon2_delete_cuda(
    ::poseidon2::Poseidon2<::bn254::scalar_t>* poseidon);
#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_ICICLE_ICICLE_POSEIDON2_BN254_H_