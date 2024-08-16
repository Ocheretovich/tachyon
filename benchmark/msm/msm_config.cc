#include "benchmark/msm/msm_config.h"

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/base/console/iostream.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/flag/flag_parser.h"
#include "tachyon/base/ranges/algorithm.h"

namespace tachyon {

using MSMConfig = benchmark::MSMConfig;

namespace base {

template <>
class FlagValueTraits<MSMConfig::TestSet> {
 public:
  static bool ParseValue(std::string_view input, MSMConfig::TestSet* value,
                         std::string* reason) {
    if (input == "random") {
      *value = MSMConfig::TestSet::kRandom;
    } else if (input == "non_uniform") {
      *value = MSMConfig::TestSet::kNonUniform;
    } else {
      *reason = absl::Substitute("Unknown test set: $0", input);
      return false;
    }
    return true;
  }
};

}  // namespace base

bool MSMConfig::Parse(int argc, char** argv,
                      const MSMConfig::Options& options) {
  // clang-format off
  parser_.AddFlag<base::Flag<std::vector<uint32_t>>>(&exponents_)
      .set_short_name("-k")
      .set_required()
      .set_help("Specify the exponent 'k' where the number of points to test is 2ᵏ.");
  // clang-format on
  parser_.AddFlag<base::Flag<TestSet>>(&test_set_)
      .set_long_name("--test_set")
      .set_help(
          "Testset to be benchmarked with. (supported testset: random, "
          "non_uniform)");
  if (options.include_vendors) {
    parser_.AddFlag<base::Flag<std::vector<benchmark::Vendor>>>(&vendors_)
        .set_long_name("--vendor")
        .set_help(
            "Vendors to be benchmarked with. (supported vendors: arkworks, "
            "bellman, halo2)");
  }

  if (!Config::Parse(
          argc, argv,
          {/*include_check_results=*/true, /*include_vendors=*/false})) {
    return false;
  }

  base::ranges::sort(exponents_);  // NOLINT
  return true;
}

std::vector<size_t> MSMConfig::GetPointNums() const {
  return base::Map(exponents_,
                   [](uint32_t exponent) { return size_t{1} << exponent; });
}

}  // namespace tachyon
