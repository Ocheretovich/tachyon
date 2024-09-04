#include "tachyon/math/polynomials/univariate/icicle/icicle_ntt_baby_bear.h"

#include "third_party/icicle/include/fields/id.h"
#include "third_party/icicle/src/ntt/ntt.cu.cc"  // NOLINT(build/include)

#include "tachyon/base/bit_cast.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/math/polynomials/univariate/icicle/icicle_ntt.h"

cudaError_t tachyon_babybear_initialize_domain(
    const ::babybear::scalar_t& primitive_root,
    ::device_context::DeviceContext& ctx, bool fast_twiddles_mode) {
  return ::ntt::init_domain(primitive_root, ctx, fast_twiddles_mode);
}

cudaError_t tachyon_babybear_ntt_cuda(
    const ::babybear::scalar_t* input, int size, ::ntt::NTTDir dir,
    ::ntt::NTTConfig<::babybear::scalar_t>& config,
    ::babybear::scalar_t* output) {
  return ::ntt::ntt(input, size, dir, config, output);
}

cudaError_t tachyon_babybear_release_domain(
    ::device_context::DeviceContext& ctx) {
  return ::ntt::release_domain<::babybear::scalar_t>(ctx);
}

namespace tachyon::math {

template <>
bool IcicleNTT<BabyBear>::Init(const BabyBear& group_gen,
                               const IcicleNTTOptions& options) {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif
  ::device_context::DeviceContext ctx{stream_, /*device_id=*/0, mem_pool_};
  math::BigInt<1> group_gen_big_int = group_gen.ToBigInt();
  // TODO(chokobole): We must handle these issues with domain initialization:
  // 1. It gets too slow when the domain size is 1, 2, 4, or small in general.
  //    See "vendors/circom/prover_main.cc".
  // 2. |fast_twiddles_mode| consumes a lot of memory, so we need to disable it
  //    if the ram of the GPU is not enough. See
  //    https://github.com/ingonyama-zk/icicle/blob/4fef542/icicle/include/ntt/ntt.cuh#L26-L40.
  gpuError_t error = tachyon_babybear_initialize_domain(
      reinterpret_cast<const ::babybear::scalar_t&>(group_gen_big_int), ctx,
      options.fast_twiddles_mode);
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed tachyon_babybear_initialize_domain()";
    return false;
  }
  VLOG(1) << "IcicleNTT is initialized";

  auto one = ::babybear::scalar_t::one();
  config_.reset(new ::ntt::NTTConfig<BabyBear>{
      ctx,
      *reinterpret_cast<BabyBear*>(&one),
      options.batch_size,
      options.columns_batch,
      options.ordering,
      options.are_inputs_on_device,
      options.are_outputs_on_device,
      options.is_async,
      /*ntt_algorithm=*/::ntt::NttAlgorithm::Auto,
  });
  return true;
}

template <>
bool IcicleNTT<BabyBear>::Run(::ntt::NttAlgorithm algorithm,
                              const BigInt& coset, BabyBear* inout, int size,
                              ::ntt::NTTDir dir) const {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif

  // NOTE(chokobole): Manual copy is needed even though
  // |sizeof(::babybear::scalar_t)| and |sizeof(BabyBear)| are same. This
  // is because their alignments are different. See
  // https://github.com/ingonyama-zk/icicle/blob/4fef542/icicle/include/fields/storage.cuh.
  uint32_t coset_gen = static_cast<uint32_t>(coset[0]);
  ::ntt::NTTConfig<::babybear::scalar_t> config{
      config_->ctx,
      *reinterpret_cast<babybear::scalar_t*>(&coset_gen),
      config_->batch_size,
      config_->columns_batch,
      config_->ordering,
      config_->are_inputs_on_device,
      config_->are_outputs_on_device,
      config_->is_async,
      algorithm,
  };

  gpuError_t error = tachyon_babybear_ntt_cuda(
      reinterpret_cast<const ::babybear::scalar_t*>(inout), size, dir, config,
      reinterpret_cast<::babybear::scalar_t*>(inout));
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed tachyon_babybear_ntt_cuda()";
    return false;
  }
  return true;
}

template <>
bool IcicleNTT<BabyBear>::Release() {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif

  ::device_context::DeviceContext ctx{stream_, /*device_id=*/0, mem_pool_};
  gpuError_t error = tachyon_babybear_release_domain(ctx);
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed tachyon_babybear_release_domain()";
    return false;
  }
  return true;
}

}  // namespace tachyon::math
