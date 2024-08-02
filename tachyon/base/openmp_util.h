#ifndef TACHYON_BASE_OPENMP_UTIL_H_
#define TACHYON_BASE_OPENMP_UTIL_H_

#include <algorithm>
#include <optional>
// NOTE(chokobole): There's no specific header for |std::size()|, but it causes
// an compiler error on g++-12. See
// https://en.cppreference.com/w/cpp/iterator/size.
#include <vector>

#if defined(TACHYON_HAS_OPENMP)
#include <omp.h>
#endif  // defined(TACHYON_HAS_OPENMP)

#if defined(TACHYON_HAS_OPENMP)
#define CONSTEXPR_IF_NOT_OPENMP
#define OMP_FOR _Pragma("omp for")
#define OMP_FOR_NOWAIT _Pragma("omp for nowait")
#define OMP_PARALLEL _Pragma("omp parallel")
#define OPENMP_PARALLEL_FOR(expr) _Pragma("omp parallel for") for (expr)
#define OPENMP_PARALLEL_NESTED_FOR(expr) \
  _Pragma("omp parallel for collapse(2)") for (expr)
#define OPENMP_FOR(expr) _Pragma("omp for") for (expr)
#else
#define CONSTEXPR_IF_NOT_OPENMP constexpr
#define OMP_FOR
#define OMP_FOR_NOWAIT
#define OMP_PARALLEL
#define OPENMP_PARALLEL_FOR(expr) for (expr)
#define OPENMP_PARALLEL_NESTED_FOR(expr) for (expr)
#define OPENMP_FOR(expr) for (expr)
#endif  // defined(TACHYON_HAS_OPENMP)

namespace tachyon::base {

// NOTE(chokobole): This function might return 0. You should handle this case
// carefully. See other examples where it is used.
template <typename Container>
size_t GetNumElementsPerThread(
    const Container& container, std::optional<size_t> threshold = std::nullopt,
    std::optional<size_t> num_of_threads = std::nullopt) {
#if defined(TACHYON_HAS_OPENMP)
  size_t thread_nums = num_of_threads.has_value()
                           ? num_of_threads.value()
                           : static_cast<size_t>(omp_get_max_threads());
#else
  size_t thread_nums = 1;
#endif
  size_t size = std::size(container);
  return (!threshold.has_value() || size > threshold.value())
             ? (size + thread_nums - 1) / thread_nums
             : size;
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_OPENMP_UTIL_H_
