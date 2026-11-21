#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <ocs2_core/constraint/StateConstraint.h>

namespace ocs2::humanoid {

/** Keeps both knees away from the straight-leg kinematic singularity. */
class KneeFlexionConstraint final : public StateConstraint {
 public:
  KneeFlexionConstraint(std::array<size_t, 2> stateIndices, scalar_t minimumFlexion)
      : StateConstraint(ConstraintOrder::Linear), indices_(stateIndices), minimumFlexion_(minimumFlexion) {
    if (!std::isfinite(minimumFlexion) || minimumFlexion <= 0.0 || indices_[0] == indices_[1]) {
      throw std::invalid_argument("Invalid knee flexion bound or state indices");
    }
  }
  KneeFlexionConstraint* clone() const override { return new KneeFlexionConstraint(*this); }
  size_t getNumConstraints(scalar_t /*time*/) const override { return 2; }
  vector_t getValue(scalar_t /*time*/, const vector_t& state, const PreComputation& /*preComp*/) const override {
    vector_t value(2);
    for (size_t i = 0; i < 2; ++i) value(i) = state(indices_[i]) - minimumFlexion_;
    return value;
  }
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time, const vector_t& state,
                                                           const PreComputation& preComp) const override {
    auto result = VectorFunctionLinearApproximation::Zero(2, state.size(), 0);
    result.f = getValue(time, state, preComp);
    for (size_t i = 0; i < 2; ++i) result.dfdx(i, indices_[i]) = 1.0;
    return result;
  }
 private:
  std::array<size_t, 2> indices_;
  scalar_t minimumFlexion_;
};

}  // namespace ocs2::humanoid
