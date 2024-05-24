#ifndef VENDORS_CIRCOM_BENCHMARK_TACHYON_RUNNER_H_
#define VENDORS_CIRCOM_BENCHMARK_TACHYON_RUNNER_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

// clang-format off
#include "benchmark/runner.h"
// clang-format on
#include "circomlib/circuit/quadratic_arithmetic_program.h"
#include "circomlib/circuit/witness_loader.h"
#include "circomlib/zkey/zkey.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/time/time.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"
#include "tachyon/zk/r1cs/groth16/prove.h"
#include "tachyon/zk/r1cs/groth16/verify.h"

namespace tachyon::circom {

template <typename Curve, size_t MaxDegree>
class TachyonRunner : public Runner<Curve> {
 public:
  using F = typename Curve::G1Curve::ScalarField;

  explicit TachyonRunner(const base::FilePath& data_path)
      : witness_loader_(data_path) {}

  WitnessLoader<F>& witness_loader() { return witness_loader_; }

  const zk::r1cs::groth16::ProvingKey<Curve>& proving_key() const {
    return proving_key_;
  }

  const zk::r1cs::ConstraintMatrices<F>& constraint_matrices() const {
    return constraint_matrices_;
  }

  void LoadZkey(const base::FilePath& zkey_path) override {
    std::unique_ptr<ZKey<Curve>> zkey = ParseZKey<Curve>(zkey_path);
    CHECK(zkey);

    proving_key_ = std::move(*zkey).TakeProvingKey().ToNativeProvingKey();
    constraint_matrices_ = std::move(*zkey).TakeConstraintMatrices().ToNative();
  }

  zk::r1cs::groth16::Proof<Curve> Run(const std::vector<F>& full_assignments,
                                      absl::Span<const F> public_inputs,
                                      base::TimeDelta& delta) override {
    using Domain = math::UnivariateEvaluationDomain<F, MaxDegree>;

    base::TimeTicks now = base::TimeTicks::Now();

    std::unique_ptr<Domain> domain =
        Domain::Create(constraint_matrices_.num_constraints +
                       constraint_matrices_.num_instance_variables);
    std::vector<F> h_evals =
        QuadraticArithmeticProgram<F>::WitnessMapFromMatrices(
            domain.get(), constraint_matrices_, full_assignments);

    LOG(ERROR) << "Witness map: " << base::TimeTicks::Now() - now;

    zk::r1cs::groth16::Proof<Curve> proof =
        zk::r1cs::groth16::CreateProofWithAssignmentNoZK(
            proving_key_, absl::MakeConstSpan(h_evals),
            absl::MakeConstSpan(full_assignments)
                .subspan(1, constraint_matrices_.num_instance_variables - 1),
            absl::MakeConstSpan(full_assignments)
                .subspan(constraint_matrices_.num_instance_variables),
            absl::MakeConstSpan(full_assignments).subspan(1));

    delta = base::TimeTicks::Now() - now;

    if (!prepared_verifying_key_.has_value()) {
      prepared_verifying_key_ =
          std::move(proving_key_).TakeVerifyingKey().ToPreparedVerifyingKey();
    }
    CHECK(zk::r1cs::groth16::VerifyProof(*prepared_verifying_key_, proof,
                                         public_inputs));

    return proof;
  }

 private:
  WitnessLoader<F> witness_loader_;
  zk::r1cs::groth16::ProvingKey<Curve> proving_key_;
  zk::r1cs::ConstraintMatrices<F> constraint_matrices_;
  std::optional<zk::r1cs::groth16::PreparedVerifyingKey<Curve>>
      prepared_verifying_key_;
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_BENCHMARK_TACHYON_RUNNER_H_
