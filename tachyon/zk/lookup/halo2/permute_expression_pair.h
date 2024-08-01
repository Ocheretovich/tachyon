// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_HALO2_PERMUTE_EXPRESSION_PAIR_H_
#define TACHYON_ZK_LOOKUP_HALO2_PERMUTE_EXPRESSION_PAIR_H_

#include <algorithm>
#include <memory_resource>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "third_party/pdqsort/include/pdqsort.h"

#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/lookup/pair.h"

namespace tachyon::zk::lookup::halo2 {

// Given a vector of input values A and a vector of table values S,
// this method permutes A and S to produce A' and S', such that:
// - like values in A' are vertically adjacent to each other; and
// - the first row in a sequence of like values in A' is the row
//   that has the corresponding value in S'.
// This method returns (A', S') if no errors are encountered.
template <typename PCS, typename Evals, typename F = typename Evals::Field>
[[nodiscard]] bool PermuteExpressionPair(ProverBase<PCS>* prover,
                                         const Pair<Evals>& in,
                                         Pair<Evals>* out) {
  size_t domain_size = prover->domain()->size();
  RowIndex usable_rows = prover->GetUsableRows();

  std::pmr::vector<F> permuted_input_expressions = in.input().evaluations();

  // sort input lookup expression values
  pdqsort(permuted_input_expressions.begin(),
          permuted_input_expressions.begin() + usable_rows);

  // a map of each unique element in the table expression and its count
  absl::btree_map<F, RowIndex> leftover_table_map;

  for (RowIndex i = 0; i < usable_rows; ++i) {
    const F& coeff = in.table()[i];
    // if key doesn't exist, insert the key and value 1 for the key.
    auto it = leftover_table_map.try_emplace(coeff, RowIndex{1});
    // no inserted value, meaning the key exists.
    if (!it.second) {
      // Increase value by 1 if not inserted.
      ++((*it.first).second);
    }
  }

  std::pmr::vector<F> permuted_table_expressions(domain_size, F::Zero());

  std::vector<RowIndex> repeated_input_rows;
  repeated_input_rows.reserve(usable_rows - 1);
  for (RowIndex row = 0; row < usable_rows; ++row) {
    const F& input_value = permuted_input_expressions[row];

    // ref: https://zcash.github.io/halo2/design/proving-system/lookup.html
    //
    // Lookup Argument must satisfy these 2 constraints.
    //
    // - constraint 1: l_first(X) * (A'(X) - S'(x)) = 0
    // - constraint 2: (A'(X) - S'(x)) * (A'(X) - A'(ω⁻¹X)) = 0
    //
    // - What 'row == 0' condition means: l_first(x) == 1.
    // To satisfy constraint 1, A'(x) - S'(x) must be 0.
    // => checking if A'(x) == S'(x)
    // - What 'input_value != permuted_input_expressions[row-1]' condition
    //   means: (A'(x) - A'(ω⁻¹x)) != 0.
    // To satisfy constraint 2, A'(x) - S'(x) must be 0.
    // => checking if A'(x) == S'(x)
    //
    // Example
    //
    // Assume that
    //  * in.input.evaluations() = [1,2,1,5]
    //  * in.table.evaluations() = [1,2,4,5]
    //
    // Result after for loop
    //
    //                   A'                      S'
    //               --------                --------
    //              |    1   |              |    1   |
    //               --------                --------
    //              |    1   |              |    4   |
    //               --------                --------
    //              |    2   |              |    2   |
    //               --------                --------
    //              |    5   |              |    5   |
    //               --------                --------
    // we can see that elements of A' {1,2,5} is in S' {1,4,2,5}
    //
    if (row == 0 || input_value != permuted_input_expressions[row - 1]) {
      // Assign S'(x) with A'(x).
      permuted_table_expressions[row] = input_value;

      // remove one instance of input_value from |leftover_table_map|.
      auto it = leftover_table_map.find(input_value);
      // if input value is not found, return error
      if (it == leftover_table_map.end()) {
        LOG(ERROR) << "input(" << input_value.ToString()
                   << ") is not found in table";
        return false;
      }

      // input value found, check if the value > 0.
      // then decrement the value by 1
      CHECK_GT(it->second--, RowIndex{0});
    } else {
      repeated_input_rows.push_back(row);
    }
  }

  // populate permuted table at unfilled rows with leftover table elements
  for (auto it = leftover_table_map.begin(); it != leftover_table_map.end();
       ++it) {
    const F& coeff = it->first;
    const RowIndex count = it->second;

    for (RowIndex i = 0; i < count; ++i) {
      CHECK(!repeated_input_rows.empty());
      RowIndex row = repeated_input_rows.back();
      permuted_table_expressions[row] = coeff;
      repeated_input_rows.pop_back();
    }
  }

  CHECK(repeated_input_rows.empty());

  Evals input(std::move(permuted_input_expressions));
  Evals table(std::move(permuted_table_expressions));

  prover->blinder().Blind(input, /*include_last_row=*/true);
  prover->blinder().Blind(table, /*include_last_row=*/true);

  *out = {std::move(input), std::move(table)};
  return true;
}

}  // namespace tachyon::zk::lookup::halo2

#endif  // TACHYON_ZK_LOOKUP_HALO2_PERMUTE_EXPRESSION_PAIR_H_
