#ifndef TACHYON_ZK_PLONK_HALO2_PROOF_H_
#define TACHYON_ZK_PLONK_HALO2_PROOF_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tachyon/base/json/json.h"
#include "tachyon/zk/lookup/halo2/verifier_data.h"
#include "tachyon/zk/lookup/log_derivative_halo2/verifier_data.h"
#include "tachyon/zk/lookup/pair.h"
#include "tachyon/zk/lookup/type.h"
#include "tachyon/zk/plonk/permutation/permutation_verifier_data.h"
#include "tachyon/zk/plonk/vanishing/vanishing_verifier_data.h"
#include "tachyon/zk/shuffle/verifier_data.h"

namespace tachyon {
namespace zk::plonk::halo2 {

template <typename F, typename C>
struct Proof {
  std::vector<std::vector<C>> advices_commitments_vec;
  std::vector<F> challenges;
  F theta = F::Zero();
  F beta = F::Zero();
  F gamma = F::Zero();
  std::vector<std::vector<C>> permutation_product_commitments_vec;
  C vanishing_random_poly_commitment;
  F y = F::Zero();
  std::vector<C> vanishing_h_poly_commitments;
  F x = F::Zero();
  std::vector<std::vector<F>> instance_evals_vec;
  std::vector<std::vector<F>> advice_evals_vec;
  std::vector<F> fixed_evals;
  F vanishing_random_eval = F::Zero();
  std::vector<F> common_permutation_evals;
  std::vector<std::vector<F>> permutation_product_evals_vec;
  std::vector<std::vector<F>> permutation_product_next_evals_vec;
  std::vector<std::vector<std::optional<F>>> permutation_product_last_evals_vec;
  std::vector<std::vector<C>> shuffle_product_commitments_vec;
  std::vector<std::vector<F>> shuffle_product_evals_vec;
  std::vector<std::vector<F>> shuffle_product_next_evals_vec;

  // auxiliary values
  F l_first = F::Zero();
  F l_blind = F::Zero();
  F l_last = F::Zero();
  F x_next = F::Zero();
  F x_prev = F::Zero();
  F x_last = F::Zero();
  F x_n = F::Zero();

  bool operator==(const Proof& other) const {
    return advices_commitments_vec == other.advices_commitments_vec &&
           challenges == other.challenges && theta == other.theta &&
           beta == other.beta && gamma == other.gamma &&
           permutation_product_commitments_vec ==
               other.permutation_product_commitments_vec &&
           vanishing_random_poly_commitment ==
               other.vanishing_random_poly_commitment &&
           y == other.y &&
           vanishing_h_poly_commitments == other.vanishing_h_poly_commitments &&
           x == other.x && instance_evals_vec == other.instance_evals_vec &&
           advice_evals_vec == other.advice_evals_vec &&
           fixed_evals == other.fixed_evals &&
           vanishing_random_eval == other.vanishing_random_eval &&
           common_permutation_evals == other.common_permutation_evals &&
           permutation_product_evals_vec ==
               other.permutation_product_evals_vec &&
           permutation_product_next_evals_vec ==
               other.permutation_product_next_evals_vec &&
           permutation_product_last_evals_vec ==
               other.permutation_product_last_evals_vec &&
           shuffle_product_commitments_vec ==
               other.shuffle_product_commitments_vec &&
           shuffle_product_evals_vec == other.shuffle_product_evals_vec &&
           shuffle_product_next_evals_vec ==
               other.shuffle_product_next_evals_vec;
  }
  bool operator!=(const Proof& other) const { return !operator==(other); }

  VanishingVerifierData<F, C> ToVanishingVerifierData(
      size_t circuit_idx, absl::Span<const C> fixed_commitments,
      absl::Span<const C> instance_commitments) const {
    return {
        fixed_commitments,
        advices_commitments_vec[circuit_idx],
        instance_commitments,
        fixed_evals,
        advice_evals_vec[circuit_idx],
        instance_evals_vec[circuit_idx],
        challenges,
        vanishing_h_poly_commitments,
        vanishing_random_poly_commitment,
        vanishing_random_eval,
    };
  }

  PermutationVerifierData<F, C> ToPermutationVerifierData(
      size_t circuit_idx,
      absl::Span<const C> common_permutation_commitments) const {
    return {
        fixed_evals,
        advice_evals_vec[circuit_idx],
        instance_evals_vec[circuit_idx],
        challenges,
        permutation_product_commitments_vec[circuit_idx],
        permutation_product_evals_vec[circuit_idx],
        permutation_product_next_evals_vec[circuit_idx],
        permutation_product_last_evals_vec[circuit_idx],
        common_permutation_commitments,
        common_permutation_evals,
        beta,
        gamma,
    };
  }

  shuffle::VerifierData<F, C> ToShuffleVerifierData(size_t circuit_idx) const {
    return {
        fixed_evals,
        advice_evals_vec[circuit_idx],
        instance_evals_vec[circuit_idx],
        challenges,
        shuffle_product_commitments_vec[circuit_idx],
        shuffle_product_evals_vec[circuit_idx],
        shuffle_product_next_evals_vec[circuit_idx],
        theta,
        gamma,
    };
  }
};

template <typename F, typename C>
struct Halo2Proof : public Proof<F, C> {
  std::vector<std::vector<lookup::Pair<C>>> lookup_permuted_commitments_vec;
  std::vector<std::vector<C>> lookup_product_commitments_vec;
  std::vector<std::vector<F>> lookup_product_evals_vec;
  std::vector<std::vector<F>> lookup_product_next_evals_vec;
  std::vector<std::vector<F>> lookup_permuted_input_evals_vec;
  std::vector<std::vector<F>> lookup_permuted_input_prev_evals_vec;
  std::vector<std::vector<F>> lookup_permuted_table_evals_vec;

  bool operator==(const Halo2Proof& other) const {
    return Proof<F, C>::operator==(other) &&
           lookup_permuted_commitments_vec ==
               other.lookup_permuted_commitments_vec &&
           lookup_product_commitments_vec ==
               other.lookup_product_commitments_vec &&
           lookup_product_evals_vec == other.lookup_product_evals_vec &&
           lookup_product_next_evals_vec ==
               other.lookup_product_next_evals_vec &&
           lookup_permuted_input_evals_vec ==
               other.lookup_permuted_input_evals_vec &&
           lookup_permuted_input_prev_evals_vec ==
               other.lookup_permuted_input_prev_evals_vec &&
           lookup_permuted_table_evals_vec ==
               other.lookup_permuted_table_evals_vec;
  }
  bool operator!=(const Halo2Proof& other) const { return !operator==(other); }

  lookup::halo2::VerifierData<F, C> ToLookupVerifierData(
      size_t circuit_idx) const {
    return {
        this->fixed_evals,
        this->advice_evals_vec[circuit_idx],
        this->instance_evals_vec[circuit_idx],
        this->challenges,
        lookup_permuted_commitments_vec[circuit_idx],
        lookup_product_commitments_vec[circuit_idx],
        lookup_product_evals_vec[circuit_idx],
        lookup_product_next_evals_vec[circuit_idx],
        lookup_permuted_input_evals_vec[circuit_idx],
        lookup_permuted_input_prev_evals_vec[circuit_idx],
        lookup_permuted_table_evals_vec[circuit_idx],
        this->theta,
        this->beta,
        this->gamma,
    };
  }
};

template <typename F, typename C>
struct LogDerivativeHalo2Proof : public Proof<F, C> {
  std::vector<std::vector<C>> lookup_m_poly_commitments_vec;
  std::vector<std::vector<C>> lookup_sum_commitments_vec;
  std::vector<std::vector<F>> lookup_sum_evals_vec;
  std::vector<std::vector<F>> lookup_sum_next_evals_vec;
  std::vector<std::vector<F>> lookup_m_evals_vec;

  bool operator==(const LogDerivativeHalo2Proof& other) const {
    return Proof<F, C>::operator==(other) &&
           lookup_m_poly_commitments_vec ==
               other.lookup_m_poly_commitments_vec &&
           lookup_sum_commitments_vec == other.lookup_sum_commitments_vec &&
           lookup_sum_evals_vec == other.lookup_sum_evals_vec &&
           lookup_sum_next_evals_vec == other.lookup_sum_next_evals_vec &&
           lookup_m_evals_vec == other.lookup_m_evals_vec;
  }
  bool operator!=(const LogDerivativeHalo2Proof& other) const {
    return !operator==(other);
  }

  lookup::log_derivative_halo2::VerifierData<F, C> ToLookupVerifierData(
      size_t circuit_idx) const {
    return {
        this->fixed_evals,
        this->advice_evals_vec[circuit_idx],
        this->instance_evals_vec[circuit_idx],
        this->challenges,
        lookup_m_poly_commitments_vec[circuit_idx],
        lookup_sum_commitments_vec[circuit_idx],
        lookup_sum_evals_vec[circuit_idx],
        lookup_sum_next_evals_vec[circuit_idx],
        lookup_m_evals_vec[circuit_idx],
        this->theta,
        this->beta,
    };
  }
};

}  // namespace zk::plonk::halo2

namespace base {

template <typename F, typename C>
class RapidJsonValueConverter<zk::plonk::halo2::Proof<F, C>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const zk::plonk::halo2::Proof<F, C>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "advices_commitments_vec",
                   value.advices_commitments_vec, allocator);
    AddJsonElement(object, "challenges", value.challenges, allocator);
    AddJsonElement(object, "theta", value.theta, allocator);
    AddJsonElement(object, "beta", value.beta, allocator);
    AddJsonElement(object, "gamma", value.gamma, allocator);
    AddJsonElement(object, "permutation_product_commitments_vec",
                   value.permutation_product_commitments_vec, allocator);
    AddJsonElement(object, "vanishing_random_poly_commitment",
                   value.vanishing_random_poly_commitment, allocator);
    AddJsonElement(object, "y", value.y, allocator);
    AddJsonElement(object, "vanishing_h_poly_commitments",
                   value.vanishing_h_poly_commitments, allocator);
    AddJsonElement(object, "x", value.x, allocator);
    AddJsonElement(object, "instance_evals_vec", value.instance_evals_vec,
                   allocator);
    AddJsonElement(object, "advice_evals_vec", value.advice_evals_vec,
                   allocator);
    AddJsonElement(object, "fixed_evals", value.fixed_evals, allocator);
    AddJsonElement(object, "vanishing_random_eval", value.vanishing_random_eval,
                   allocator);
    AddJsonElement(object, "common_permutation_evals",
                   value.common_permutation_evals, allocator);
    AddJsonElement(object, "permutation_product_evals_vec",
                   value.permutation_product_evals_vec, allocator);
    AddJsonElement(object, "permutation_product_next_evals_vec",
                   value.permutation_product_next_evals_vec, allocator);
    AddJsonElement(object, "permutation_product_last_evals_vec",
                   value.permutation_product_last_evals_vec, allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 zk::plonk::halo2::Proof<F, C>* proof_out, std::string* error) {
    zk::plonk::halo2::Proof<F, C> proof;
    if (!ParseJsonElement(json_value, "advices_commitments_vec",
                          &proof.advices_commitments_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "challenges", &proof.challenges, error))
      return false;
    if (!ParseJsonElement(json_value, "theta", &proof.theta, error))
      return false;
    if (!ParseJsonElement(json_value, "beta", &proof.beta, error)) return false;
    if (!ParseJsonElement(json_value, "gamma", &proof.gamma, error))
      return false;
    if (!ParseJsonElement(json_value, "permutation_product_commitments_vec",
                          &proof.permutation_product_commitments_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "vanishing_random_poly_commitment",
                          &proof.vanishing_random_poly_commitment, error))
      return false;
    if (!ParseJsonElement(json_value, "y", &proof.y, error)) return false;
    if (!ParseJsonElement(json_value, "vanishing_h_poly_commitments",
                          &proof.vanishing_h_poly_commitments, error))
      return false;
    if (!ParseJsonElement(json_value, "x", &proof.x, error)) return false;
    if (!ParseJsonElement(json_value, "instance_evals_vec",
                          &proof.instance_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "advice_evals_vec",
                          &proof.advice_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "fixed_evals", &proof.fixed_evals, error))
      return false;
    if (!ParseJsonElement(json_value, "vanishing_random_eval",
                          &proof.vanishing_random_eval, error))
      return false;
    if (!ParseJsonElement(json_value, "common_permutation_evals",
                          &proof.common_permutation_evals, error))
      return false;
    if (!ParseJsonElement(json_value, "permutation_product_evals_vec",
                          &proof.permutation_product_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "permutation_product_next_evals_vec",
                          &proof.permutation_product_next_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "common_permutation_evals",
                          &proof.common_permutation_evals, error))
      return false;
    if (!ParseJsonElement(json_value, "permutation_product_evals_vec",
                          &proof.permutation_product_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "permutation_product_next_evals_vec",
                          &proof.permutation_product_next_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "permutation_product_last_evals_vec",
                          &proof.permutation_product_last_evals_vec, error))
      return false;

    *proof_out = std::move(proof);
    return true;
  }
};

template <typename F, typename C>
class RapidJsonValueConverter<zk::plonk::halo2::Halo2Proof<F, C>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const zk::plonk::halo2::Halo2Proof<F, C>& value,
                               Allocator& allocator) {
    rapidjson::Value object =
        RapidJsonValueConverter<zk::plonk::halo2::Proof<F, C>>::From(value,
                                                                     allocator);
    AddJsonElement(object, "lookup_permuted_commitments_vec",
                   value.lookup_permuted_commitments_vec, allocator);
    AddJsonElement(object, "lookup_product_commitments_vec",
                   value.lookup_product_commitments_vec, allocator);
    AddJsonElement(object, "lookup_product_evals_vec",
                   value.lookup_product_evals_vec, allocator);
    AddJsonElement(object, "lookup_product_next_evals_vec",
                   value.lookup_product_next_evals_vec, allocator);
    AddJsonElement(object, "lookup_permuted_input_evals_vec",
                   value.lookup_permuted_input_evals_vec, allocator);
    AddJsonElement(object, "lookup_permuted_input_prev_evals_vec",
                   value.lookup_permuted_input_prev_evals_vec, allocator);
    AddJsonElement(object, "lookup_permuted_table_evals_vec",
                   value.lookup_permuted_table_evals_vec, allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 zk::plonk::halo2::Halo2Proof<F, C>* proof_out,
                 std::string* error) {
    zk::plonk::halo2::Halo2Proof<F, C> proof;
    if (!RapidJsonValueConverter<zk::plonk::halo2::Proof<F, C>>::To(
            json_value, key, &proof, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_permuted_commitments_vec",
                          &proof.lookup_permuted_commitments_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_product_commitments_vec",
                          &proof.lookup_product_commitments_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_product_evals_vec",
                          &proof.lookup_product_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_product_next_evals_vec",
                          &proof.lookup_product_next_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_permuted_input_evals_vec",
                          &proof.lookup_permuted_input_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_permuted_input_prev_evals_vec",
                          &proof.lookup_permuted_input_prev_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_permuted_table_evals_vec",
                          &proof.lookup_permuted_table_evals_vec, error))
      return false;

    *proof_out = std::move(proof);
    return true;
  }
};

template <typename F, typename C>
class RapidJsonValueConverter<zk::plonk::halo2::LogDerivativeHalo2Proof<F, C>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(
      const zk::plonk::halo2::LogDerivativeHalo2Proof<F, C>& value,
      Allocator& allocator) {
    rapidjson::Value object =
        RapidJsonValueConverter<zk::plonk::halo2::Proof<F, C>>::From(value,
                                                                     allocator);
    AddJsonElement(object, "lookup_m_poly_commitments_vec",
                   value.lookup_m_poly_commitments_vec, allocator);
    AddJsonElement(object, "lookup_sum_commitments_vec",
                   value.lookup_sum_commitments_vec, allocator);
    AddJsonElement(object, "lookup_sum_evals_vec", value.lookup_sum_evals_vec,
                   allocator);
    AddJsonElement(object, "lookup_sum_next_evals_vec",
                   value.lookup_sum_next_evals_vec, allocator);
    AddJsonElement(object, "lookup_m_evals_vec", value.lookup_m_evals_vec,
                   allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 zk::plonk::halo2::LogDerivativeHalo2Proof<F, C>* proof_out,
                 std::string* error) {
    zk::plonk::halo2::LogDerivativeHalo2Proof<F, C> proof;
    if (!RapidJsonValueConverter<zk::plonk::halo2::Proof<F, C>>::To(
            json_value, key, &proof, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_m_poly_commitments_vec",
                          &proof.lookup_m_poly_commitments_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_sum_commitments_vec",
                          &proof.lookup_sum_commitments_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_sum_evals_vec",
                          &proof.lookup_sum_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_sum_next_evals_vec",
                          &proof.lookup_sum_next_evals_vec, error))
      return false;
    if (!ParseJsonElement(json_value, "lookup_m_evals_vec",
                          &proof.lookup_m_evals_vec, error))
      return false;

    *proof_out = std::move(proof);
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_HALO2_PROOF_H_
