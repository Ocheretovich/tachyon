// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_PROOF_SERIALIZER_H_
#define TACHYON_ZK_PLONK_HALO2_PROOF_SERIALIZER_H_

#include <type_traits>
#include <utility>

#include "tachyon/base/buffer/buffer.h"
#include "tachyon/math/elliptic_curves/affine_point.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon::zk::plonk::halo2 {

template <typename F, typename SFINAE = void>
class ProofSerializer;

template <typename F>
class ProofSerializer<
    F, std::enable_if_t<std::is_base_of_v<math::PrimeFieldBase<F>, F>>> {
 public:
  [[nodiscard]] static bool ReadFromProof(const base::ReadOnlyBuffer& buffer,
                                          F* scalar) {
    return buffer.Read(scalar);
  }

  [[nodiscard]] static bool WriteToProof(const F& scalar,
                                         base::Buffer& buffer) {
    return buffer.Write(scalar);
  }
};

template <typename Curve>
class ProofSerializer<math::AffinePoint<Curve>> {
 public:
  using BaseField = typename math::AffinePoint<Curve>::BaseField;
  using BigInt = typename BaseField::BigIntTy;

  static_assert(BaseField::kModulusBits % 8 != 0,
                "Halo2 needs 2 spare bits to put flags");

  constexpr static size_t kOriginalByteSize =
      BaseField::kLimbNums * sizeof(uint64_t);
  constexpr static size_t kByteSize =
      ((kOriginalByteSize * 8 - BaseField::kModulusBits) > 1)
          ? kOriginalByteSize
          : kOriginalByteSize + 1;

  [[nodiscard]] static bool ReadFromProof(const base::ReadOnlyBuffer& buffer,
                                          math::AffinePoint<Curve>* point_out) {
    uint8_t bytes[kByteSize];
    if (!buffer.Read(bytes)) return false;

    uint8_t is_inf = bytes[kByteSize - 1] >> 7;
    uint8_t y_sign = (bytes[kByteSize - 1] >> 6) & 1;
    bytes[kByteSize - 1] &= 0b00111111;
    BaseField x = BaseField::FromBigInt(BigInt::FromBytesLE(bytes));
    if (x.IsZero() && is_inf) {
      *point_out = math::AffinePoint<Curve>::Zero();
      return true;
    } else {
      std::optional<math::AffinePoint<Curve>> point =
          math::AffinePoint<Curve>::CreateFromX(x, y_sign);
      if (!point.has_value()) return false;
      *point_out = std::move(point).value();
      return true;
    }
  }

  [[nodiscard]] static bool WriteToProof(const math::AffinePoint<Curve>& point,
                                         base::Buffer& buffer) {
    if (point.infinity()) {
      uint8_t infinity_bytes[kByteSize] = {0};
      infinity_bytes[kByteSize - 1] |= 0b10000000;
      return buffer.Write(infinity_bytes);
    } else {
      uint8_t y_sign = uint8_t{point.y().ToBigInt().IsOdd()} << 6;
      std::array<uint8_t, kByteSize> x = point.x().ToBigInt().ToBytesLE();
      if (!buffer.Write(x)) return false;
      return buffer.WriteAt(buffer.buffer_offset() - 1,
                            static_cast<uint8_t>(x[kByteSize - 1] | y_sign));
    }
  }
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_PROOF_SERIALIZER_H_
