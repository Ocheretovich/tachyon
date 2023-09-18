#if TACHYON_CUDA || TACHYON_USE_ROCM
#include <iostream>

// clang-format off
#include "benchmark/msm/msm_config.h"
#include "benchmark/msm/msm_runner.h"
#include "benchmark/msm/simple_msm_benchmark_reporter.h"
// clang-format on
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/msm/msm.h"
#include "tachyon/c/math/elliptic_curves/msm/msm_gpu.h"
#include "tachyon/math/elliptic_curves/test/random.h"

namespace tachyon {

using namespace math;

int RealMain(int argc, char** argv) {
  MSMConfig config;
  if (!config.Parse(argc, argv, false)) {
    return 1;
  }

  SimpleMSMBenchmarkReporter reporter("MSM Benchmark GPU", config.degrees());
  reporter.AddVendor("tachyon_gpu");

  std::vector<uint64_t> point_nums = config.GetPointNums();

  tachyon_init_msm(config.degrees().back());

  std::cout << "Generating random points..." << std::endl;
  uint64_t max_point_num = point_nums.back();
  std::vector<bn254::G1AffinePoint> bases =
      CreatePseudoRandomPoints<bn254::G1AffinePoint>(max_point_num);
  std::vector<bn254::Fr> scalars =
      base::CreateVector(max_point_num, []() { return bn254::Fr::Random(); });
  std::cout << "Generation completed" << std::endl;

  MSMRunner<bn254::G1AffinePoint> runner(&reporter);
  runner.SetInputs(&bases, &scalars);

  std::vector<bn254::G1JacobianPoint> results_cpu;
  runner.Run(tachyon_bn254_g1_affine_msm, point_nums, &results_cpu);
  tachyon_release_msm();

  tachyon_init_msm_gpu(config.degrees().back());
  std::vector<bn254::G1JacobianPoint> results_gpu;
  runner.Run(tachyon_bn254_g1_affine_msm_gpu, point_nums, &results_gpu);
  tachyon_release_msm_gpu();

  CHECK(results_cpu == results_gpu) << "Result not matched";

  reporter.Show();

  return 0;
}

}  // namespace tachyon

int main(int argc, char** argv) { return tachyon::RealMain(argc, argv); }
#else
#include "tachyon/base/console/iostream.h"

int main(int argc, char **argv) {
  tachyon_cerr << "please build with --config cuda or --config rocm"
               << std::endl;
  return 1;
}
#endif  // TACHYON_CUDA