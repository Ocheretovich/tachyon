#ifndef VENDORS_CIRCOM_BENCHMARK_TACHYON_RUNNER_H_
#define VENDORS_CIRCOM_BENCHMARK_TACHYON_RUNNER_H_

#include <memory>
#include <optional>
#include <vector>

#include "tachyon/base/memory/reusing_allocator.h"

// clang-format off
#include "benchmark/runner.h"
// clang-format on
#include "circomlib/circuit/quadratic_arithmetic_program.h"
#include "circomlib/circuit/witness_loader.h"
#include "circomlib/zkey/zkey.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/time/time.h"
#include "tachyon/zk/r1cs/groth16/prove.h"
#include "tachyon/zk/r1cs/groth16/verify.h"

namespace tachyon::circom {

template <typename Curve, size_t MaxDegree>
class TachyonRunner : public Runner<Curve, MaxDegree> {
 public:
  using F = typename Curve::G1Curve::ScalarField;
  using Domain = math::UnivariateEvaluationDomain<F, MaxDegree>;

  explicit TachyonRunner(const base::FilePath& data_path)
      : witness_loader_(data_path) {}

  WitnessLoader<F>& witness_loader() { return witness_loader_; }

  const zk::r1cs::groth16::ProvingKey<Curve>& proving_key() const {
    return proving_key_;
  }

  size_t GetDomainSize() const { return zkey_->GetDomainSize(); }

  size_t GetNumInstanceVariables() const {
    return zkey_->GetNumInstanceVariables();
  }

  size_t GetNumWitnessVariables() const {
    return zkey_->GetNumWitnessVariables();
  }

  void LoadZKey(const base::FilePath& zkey_path) override {
    zkey_ = ParseZKey<Curve>(zkey_path);
    CHECK(zkey_);

    proving_key_ = zkey_->GetProvingKey().ToNativeProvingKey();
    coefficients_ = zkey_->GetCoefficients();
  }

  zk::r1cs::groth16::Proof<Curve> Run(const Domain* domain,
                                      const std::vector<F>& full_assignments,
                                      absl::Span<const F> public_inputs,
                                      base::TimeDelta& delta) override {
    base::TimeTicks now = base::TimeTicks::Now();

    std::vector<F, base::memory::ReusingAllocator<F>> h_evals =
        QuadraticArithmeticProgram<F>::WitnessMapFromMatrices(
            domain, coefficients_, full_assignments);

    size_t num_instance_variables = GetNumInstanceVariables();
    zk::r1cs::groth16::Proof<Curve> proof =
        zk::r1cs::groth16::CreateProofWithAssignmentNoZK(
            proving_key_, absl::MakeConstSpan(h_evals),
            absl::MakeConstSpan(full_assignments)
                .subspan(1, num_instance_variables - 1),
            absl::MakeConstSpan(full_assignments)
                .subspan(num_instance_variables),
            absl::MakeConstSpan(full_assignments).subspan(1));

    delta = base::TimeTicks::Now() - now;

    if (!prepared_verifying_key_.has_value()) {
      prepared_verifying_key_ =
          proving_key_.verifying_key().ToPreparedVerifyingKey();
    }
    CHECK(zk::r1cs::groth16::VerifyProof(*prepared_verifying_key_, proof,
                                         public_inputs));

    return proof;
  }

 private:
  WitnessLoader<F> witness_loader_;
  std::unique_ptr<ZKey<Curve>> zkey_;
  zk::r1cs::groth16::ProvingKey<Curve> proving_key_;
  absl::Span<const Coefficient<F>> coefficients_;
  std::optional<zk::r1cs::groth16::PreparedVerifyingKey<Curve>>
      prepared_verifying_key_;
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_BENCHMARK_TACHYON_RUNNER_H_
