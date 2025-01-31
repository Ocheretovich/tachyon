// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_QUADRATIC_ARITHMETIC_PROGRAM_H_
#define VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_QUADRATIC_ARITHMETIC_PROGRAM_H_

#include <utility>
#include <vector>

#include "circomlib/zkey/coefficient.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/profiler.h"
#include "tachyon/zk/r1cs/constraint_system/quadratic_arithmetic_program.h"

namespace tachyon::circom {

template <typename F>
class QuadraticArithmeticProgram {
 public:
  QuadraticArithmeticProgram() = delete;

  template <typename Domain>
  static std::vector<F> WitnessMapFromMatrices(
      const Domain* domain, absl::Span<const Coefficient<F>> coefficients,
      absl::Span<const F> full_assignments) {
    TRACE_EVENT("ProofGeneration", "QAP::WitnessMapFromMatrices");
    using Evals = typename Domain::Evals;
    using DensePoly = typename Domain::DensePoly;

    TRACE_EVENT_BEGIN("Subtask", "ABC Initialization");
    std::vector<F> a(domain->size());
    std::vector<F> b(domain->size());
    std::vector<F> c(domain->size());
    TRACE_EVENT_END("Subtask");

    TRACE_EVENT_BEGIN("Subtask", "AB Calculation");
    // See
    // https://github.com/iden3/rapidsnark/blob/b17e6fe/src/groth16.cpp#L116-L156.
#if defined(TACHYON_HAS_OPENMP)
    constexpr size_t kNumLocks = 1024;
    omp_lock_t locks[kNumLocks];
    for (size_t i = 0; i < kNumLocks; i++) omp_init_lock(&locks[i]);
#endif
    OMP_PARALLEL_FOR(size_t i = 0; i < coefficients.size(); i++) {
      const Coefficient<F>& c = coefficients[i];
      std::vector<F>& ab = (c.matrix == 0) ? a : b;

#if defined(TACHYON_HAS_OPENMP)
      omp_set_lock(&locks[c.constraint % kNumLocks]);
#endif
      if (c.value.IsOne()) {
        ab[c.constraint] += full_assignments[c.signal];
      } else {
        ab[c.constraint] += c.value * full_assignments[c.signal];
      }
#if defined(TACHYON_HAS_OPENMP)
      omp_unset_lock(&locks[c.constraint % kNumLocks]);
#endif
    }
#if defined(TACHYON_HAS_OPENMP)
    for (size_t i = 0; i < kNumLocks; i++) omp_destroy_lock(&locks[i]);
#endif
    TRACE_EVENT_END("Subtask");

    {
      TRACE_EVENT("Subtask", "C Calculation");
      OMP_PARALLEL_FOR(size_t i = 0; i < domain->size(); ++i) {
        c[i] = a[i] * b[i];
      }
    }

    Evals a_evals(std::move(a));
    Evals b_evals(std::move(b));
    Evals c_evals(std::move(c));

    DensePoly a_poly, b_poly, c_poly;
    {
      TRACE_EVENT("Subtask", "ABC IFFT");
      a_poly = domain->IFFT(std::move(a_evals));
      b_poly = domain->IFFT(std::move(b_evals));
      c_poly = domain->IFFT(std::move(c_evals));
    }

    F root_of_unity;
    CHECK(F::GetRootOfUnity(2 * domain->size(), &root_of_unity));

    {
      TRACE_EVENT("Subtask", "ABC UpdateCosets");
      Domain::DistributePowers(a_poly, root_of_unity);
      Domain::DistributePowers(b_poly, root_of_unity);
      Domain::DistributePowers(c_poly, root_of_unity);
    }

    {
      TRACE_EVENT("Subtask", "ABC FFT");
      a_evals = domain->FFT(std::move(a_poly));
      b_evals = domain->FFT(std::move(b_poly));
      c_evals = domain->FFT(std::move(c_poly));
    }

    {
      TRACE_EVENT("Subtask", "HEval Calculation");
      // |h_evals[i]| = |a[i]| * |b[i]| - |c[i]|
      OMP_PARALLEL_FOR(size_t i = 0; i < domain->size(); ++i) {
        F& h_evals_i = a_evals.at(i);
        h_evals_i *= b_evals[i];
        h_evals_i -= c_evals[i];
      }
    }
    return std::move(a_evals).TakeEvaluations();
  }
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_CIRCUIT_QUADRATIC_ARITHMETIC_PROGRAM_H_
