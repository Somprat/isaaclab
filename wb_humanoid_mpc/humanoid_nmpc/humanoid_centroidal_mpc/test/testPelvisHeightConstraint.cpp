#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include "humanoid_centroidal_mpc/constraint/PelvisHeightConstraint.h"

using namespace ocs2;
using namespace ocs2::humanoid;

TEST(PelvisHeightConstraint, FeasibilityAndGradient) {
  PelvisHeightConstraint constraint(0.70);
  PreComputation preComp;
  vector_t state = vector_t::Zero(41);
  for (const auto height : {0.461, 0.70, 0.7925}) {
    state(8) = height;
    EXPECT_NEAR(constraint.getValue(0, state, preComp)(0), height - 0.70, 1e-12);
    const auto approximation = constraint.getLinearApproximation(0, state, preComp);
    for (int i = 0; i < state.size(); ++i) {
      auto perturbed = state;
      perturbed(i) += 1e-6;
      const auto difference = (constraint.getValue(0, perturbed, preComp)(0) - approximation.f(0)) / 1e-6;
      EXPECT_NEAR(approximation.dfdx(0, i), difference, 1e-8);
    }
  }
  std::unique_ptr<StateConstraint> copy(constraint.clone());
  EXPECT_DOUBLE_EQ(copy->getValue(0, state, preComp)(0), constraint.getValue(0, state, preComp)(0));
}

TEST(PelvisHeightConstraint, RejectInvalidFloor) {
  for (const auto height : {0.0, -0.7, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(PelvisHeightConstraint{height}, std::invalid_argument);
  }
}

#include "humanoid_centroidal_mpc/constraint/KneeFlexionConstraint.h"

TEST(KneeFlexionConstraint, BoundsAndGradient) {
  KneeFlexionConstraint constraint({15, 21}, 0.15);
  PreComputation preComp;
  auto state = vector_t::Zero(41).eval();
  state(15) = 0.1;
  state(21) = 0.3;
  const auto approximation = constraint.getLinearApproximation(0, state, preComp);
  EXPECT_NEAR(approximation.f(0), -0.05, 1e-12);
  EXPECT_NEAR(approximation.f(1), 0.15, 1e-12);
  for (int j = 0; j < state.size(); ++j) {
    auto perturbed = state;
    perturbed(j) += 1e-6;
    const vector_t difference = (constraint.getValue(0, perturbed, preComp) - approximation.f) / 1e-6;
    EXPECT_TRUE(approximation.dfdx.col(j).isApprox(difference, 1e-8));
  }
  std::unique_ptr<StateConstraint> copy(constraint.clone());
  EXPECT_TRUE(copy->getValue(0, state, preComp).isApprox(approximation.f));
  EXPECT_THROW((KneeFlexionConstraint{{15,21}, 0}), std::invalid_argument);
  EXPECT_THROW((KneeFlexionConstraint{{15,15}, .15}), std::invalid_argument);
  EXPECT_THROW((KneeFlexionConstraint{{15,21}, std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
}
