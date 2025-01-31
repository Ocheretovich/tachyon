#include <iostream>

// clang-format off
#include "benchmark/simple_reporter.h"
#include "benchmark/poseidon2/poseidon2_benchmark_runner.h"
#include "benchmark/poseidon2/poseidon2_config.h"
// clang-format on
#include "tachyon/base/containers/contains.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/profiler.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/param_traits/poseidon2_baby_bear.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/param_traits/poseidon2_bn254.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_config.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_params.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"

namespace tachyon::benchmark {

using namespace crypto;

extern "C" tachyon_baby_bear* run_poseidon2_horizen_baby_bear(
    uint64_t* duration);
extern "C" tachyon_baby_bear* run_poseidon2_plonky3_baby_bear(
    uint64_t* duration);
extern "C" tachyon_bn254_fr* run_poseidon2_horizen_bn254_fr(uint64_t* duration);
extern "C" tachyon_bn254_fr* run_poseidon2_plonky3_bn254_fr(uint64_t* duration);

template <typename Field, typename Fn>
void Run(SimpleReporter& reporter, const Poseidon2Config& config, Fn horizen_fn,
         Fn plonky3_fn) {
  Field::Init();

  Poseidon2BenchmarkRunner<Field> runner(reporter, config);

  Field result;
  if constexpr (std::is_same_v<Field, math::BabyBear>) {
    if (base::Contains(config.vendors(), Vendor::Plonky3())) {
      using Params = Poseidon2Params<math::BabyBear, 15, 7>;
      auto poseidon2_config = crypto::Poseidon2Config<Params>::Create(
          crypto::GetPoseidon2InternalShiftArray<Params>());
      result = runner.Run(poseidon2_config);
    } else {
      using Params = Poseidon2Params<math::BabyBear, 15, 7>;
      auto poseidon2_config = crypto::Poseidon2Config<Params>::Create(
          crypto::GetPoseidon2InternalDiagonalArray<Params>());
      result = runner.Run(poseidon2_config);
    }
  } else {
    using Params = Poseidon2Params<math::bn254::Fr, 2, 5>;
    auto poseidon2_config = crypto::Poseidon2Config<Params>::Create(
        crypto::GetPoseidon2InternalDiagonalArray<Params>());
    result = runner.Run(poseidon2_config);
  }
  for (const Vendor vendor : config.vendors()) {
    Field result_vendor;
    if (vendor.value() == Vendor::kHorizen) {
      result_vendor = runner.RunExternal(vendor, horizen_fn);
    } else if (vendor.value() == Vendor::kPlonky3) {
      result_vendor = runner.RunExternal(vendor, plonky3_fn);
    } else {
      NOTREACHED();
    }

    if (config.check_results()) {
      if constexpr (Field::Config::kModulusBits < 32) {
        if (vendor.value() == Vendor::kHorizen) {
          // NOTE(ashjeong): horizen's montgomery R = tachyon's montgomery R²
          CHECK_EQ(result, Field::FromMontgomery(result_vendor.ToBigInt()[0]))
              << "Tachyon and Horizen results do not match";
        } else if (vendor.value() == Vendor::kPlonky3) {
          CHECK_EQ(result, result_vendor)
              << "Tachyon and Plonky3 results do not match";
        }
      } else {
        CHECK_EQ(result, result_vendor) << "Results do not match";
      }
    }
  }
}

int RealMain(int argc, char** argv) {
  base::FilePath tmp_file;
  CHECK(base::GetTempDir(&tmp_file));
  tmp_file = tmp_file.Append("poseidon2_benchmark.perfetto-trace");
  base::Profiler profiler({tmp_file});

  profiler.Init();
  profiler.Start();

  Poseidon2Config config;
  if (!config.Parse(argc, argv)) {
    return 1;
  }

  SimpleReporter reporter;
  reporter.set_title("Poseidon2 Benchmark");
  reporter.set_x_label("Trial number");
  reporter.set_column_labels(
      base::CreateVector(config.repeating_num(),
                         [](size_t i) { return base::NumberToString(i); }));

  if (config.prime_field().value() == FieldType::kBabyBear) {
    Run<math::BabyBear>(reporter, config, run_poseidon2_horizen_baby_bear,
                        run_poseidon2_plonky3_baby_bear);
  } else if (config.prime_field().value() == FieldType::kBn254Fr) {
    Run<math::bn254::Fr>(reporter, config, run_poseidon2_horizen_bn254_fr,
                         run_poseidon2_plonky3_bn254_fr);
  } else {
    NOTREACHED();
  }

  reporter.AddAverageAsLastColumn();
  reporter.Show();

  return 0;
}

}  // namespace tachyon::benchmark

int main(int argc, char** argv) {
  return tachyon::benchmark::RealMain(argc, argv);
}
