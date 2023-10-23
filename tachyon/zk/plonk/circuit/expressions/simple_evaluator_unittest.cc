#include "tachyon/zk/plonk/circuit/expressions/simple_evaluator.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/base/logging.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"
#include "tachyon/zk/plonk/circuit/expressions/expression_factory.h"

namespace tachyon::zk {

namespace {

const size_t kMaxDegree = 5;

using GF7 = math::GF7;
using Expr = std::unique_ptr<Expression<GF7>>;
using Poly = math::UnivariateDensePolynomial<GF7, kMaxDegree>;
using Coeffs = math::UnivariateDenseCoefficients<GF7, kMaxDegree>;

class SimpleEvaluatorTest : public testing::Test {
 public:
  static void SetUpTestSuite() { math::GF7Config::Init(); }

  SimpleEvaluatorTest() {
    fixed_polys_.push_back(Poly(Coeffs({GF7(3)})));
    fixed_polys_.push_back(Poly(Coeffs({GF7(2), GF7(4)})));

    advice_polys_.push_back(Poly(Coeffs({GF7(1)})));
    advice_polys_.push_back(Poly(Coeffs({GF7(2), GF7(3)})));
    advice_polys_.push_back(Poly(Coeffs({GF7(4), GF7(5)})));
    advice_polys_.push_back(Poly(Coeffs({GF7(6), GF7(1), GF7(2)})));

    instance_polys_.push_back(Poly(Coeffs({GF7(1)})));
    instance_polys_.push_back(Poly(Coeffs({GF7(4), GF7(1)})));
    instance_polys_.push_back(Poly(Coeffs({GF7(2), GF7(3)})));
    instance_polys_.push_back(Poly(Coeffs({GF7(5), GF7(6)})));

    challenges_.push_back(GF7(1));
    challenges_.push_back(GF7(3));
    challenges_.push_back(GF7(4));
    challenges_.push_back(GF7(5));

    simple_evaluator_ = std::unique_ptr<SimpleEvaluator<Poly>>(
        new SimpleEvaluator<Poly>(3, 4, 1, &fixed_polys_, &advice_polys_,
                                  &instance_polys_, &challenges_));
  }
  SimpleEvaluatorTest(const SimpleEvaluatorTest&) = delete;
  SimpleEvaluatorTest& operator=(const SimpleEvaluatorTest&) = delete;
  ~SimpleEvaluatorTest() override = default;

 protected:
  std::vector<Poly> fixed_polys_;
  std::vector<Poly> advice_polys_;
  std::vector<Poly> instance_polys_;
  std::vector<GF7> challenges_;
  std::unique_ptr<SimpleEvaluator<Poly>> simple_evaluator_;
};

}  // namespace

TEST_F(SimpleEvaluatorTest, Constant) {
  GF7 value = GF7::Random();
  std::unique_ptr<Expression<GF7>> constant_expr =
      ExpressionFactory<GF7>::Constant(value);
  GF7 evaluated = simple_evaluator_->Evaluate(constant_expr.get());
  EXPECT_EQ(value, evaluated);
}

TEST_F(SimpleEvaluatorTest, Fixed) {
  struct {
    size_t column_index;
    int32_t rotation;
  } tests[] = {
      // fixed_polys_[0], (3 + 1 * 1) % 4 = 0, coefficient[0]
      {0, 1},
      // fixed_polys_[1], (3 + 2 * 1) % 4 = 1, coefficient[1]
      {1, 2},
  };

  for (const auto& test : tests) {
    int32_t idx = simple_evaluator_->idx();
    int32_t rot_scale = simple_evaluator_->rot_scale();
    int32_t size = simple_evaluator_->size();
    FixedQuery query(1, Rotation(test.rotation),
                     FixedColumn(test.column_index));
    size_t coeff_index = query.rotation().GetIndex(idx, rot_scale, size);

    const GF7* expected = fixed_polys_[test.column_index][coeff_index];
    Expr expr = ExpressionFactory<GF7>::Fixed(query);
    GF7 evaluated = simple_evaluator_->Evaluate(expr.get());
    EXPECT_EQ(evaluated, *expected);
  }
}

TEST_F(SimpleEvaluatorTest, Advice) {
  struct {
    size_t column_index;
    int32_t rotation;
  } tests[] = {
      // advice_polys_[2], (3 + 6 * 1) % 4 = 1 coefficient[1]
      {2, 6},
      // advice_polys_[3], (3 + 7 * 1) % 4 = 2 coefficient[2]
      {3, 7},
  };

  for (const auto& test : tests) {
    int32_t idx = simple_evaluator_->idx();
    int32_t rot_scale = simple_evaluator_->rot_scale();
    int32_t size = simple_evaluator_->size();
    AdviceQuery query(1, Rotation(test.rotation),
                      AdviceColumn(test.column_index, Phase(0)));
    size_t coeff_index = query.rotation().GetIndex(idx, rot_scale, size);

    const GF7* expected = advice_polys_[test.column_index][coeff_index];
    Expr expr = ExpressionFactory<GF7>::Advice(query);
    GF7 evaluated = simple_evaluator_->Evaluate(expr.get());
    EXPECT_EQ(evaluated, *expected);
  }
}

TEST_F(SimpleEvaluatorTest, Instance) {
  struct {
    size_t column_index;
    int32_t rotation;
  } tests[] = {
      // instance_polys_[1], (3 + 1 * 1) % 4 = 0 coefficient[0]
      {1, 1},
      // instance_polys_[2], (3 + 2 * 1) % 4 = 1 coefficient[1]
      {2, 2},
  };

  for (const auto& test : tests) {
    int32_t idx = simple_evaluator_->idx();
    int32_t rot_scale = simple_evaluator_->rot_scale();
    int32_t size = simple_evaluator_->size();
    InstanceQuery query(1, Rotation(test.rotation),
                        InstanceColumn(test.column_index));
    size_t coeff_index = query.rotation().GetIndex(idx, rot_scale, size);

    const GF7* expected = instance_polys_[test.column_index][coeff_index];
    Expr expr = ExpressionFactory<GF7>::Instance(query);
    GF7 evaluated = simple_evaluator_->Evaluate(expr.get());
    EXPECT_EQ(evaluated, *expected);
  }
}

TEST_F(SimpleEvaluatorTest, Challenges) {
  for (size_t i = 0; i < challenges_.size(); ++i) {
    Expr expr = ExpressionFactory<GF7>::Challenge(Challenge(i, Phase(0)));
    GF7 evaluated = simple_evaluator_->Evaluate(expr.get());
    GF7 expected = challenges_[i];
    EXPECT_EQ(evaluated, expected);
  }
}

TEST_F(SimpleEvaluatorTest, Negated) {
  GF7 value = GF7::Random();
  Expr negated_expression =
      ExpressionFactory<GF7>::Negated(ExpressionFactory<GF7>::Constant(value));
  GF7 negated = simple_evaluator_->Evaluate(negated_expression.get());
  EXPECT_EQ(negated, -value);
}

TEST_F(SimpleEvaluatorTest, Sum) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr sum_expression =
      ExpressionFactory<GF7>::Sum({ExpressionFactory<GF7>::Constant(a),
                                   ExpressionFactory<GF7>::Constant(b)});
  GF7 evaluated = simple_evaluator_->Evaluate(sum_expression.get());
  EXPECT_EQ(a + b, evaluated);
}

TEST_F(SimpleEvaluatorTest, Product) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr product_expression =
      ExpressionFactory<GF7>::Product({ExpressionFactory<GF7>::Constant(a),
                                       ExpressionFactory<GF7>::Constant(b)});
  GF7 evaluated = simple_evaluator_->Evaluate(product_expression.get());
  EXPECT_EQ(a * b, evaluated);
}

TEST_F(SimpleEvaluatorTest, Scaled) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr scaled_expression =
      ExpressionFactory<GF7>::Scaled({ExpressionFactory<GF7>::Constant(a), b});
  GF7 scaled_expr = simple_evaluator_->Evaluate(scaled_expression.get());
  EXPECT_EQ(scaled_expr, a * b);
}

}  // namespace tachyon::zk
