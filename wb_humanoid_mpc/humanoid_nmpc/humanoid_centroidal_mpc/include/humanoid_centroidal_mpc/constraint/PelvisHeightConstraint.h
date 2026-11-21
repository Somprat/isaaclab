#pragma once

#include <cmath>
#include <stdexcept>

#include <ocs2_core/constraint/StateConstraint.h>

namespace ocs2::humanoid {

/** Flat-ground pelvis lower bound: h(x) = pelvis_z - minimumHeight >= 0.
 * Centroidal state is [normalized momentum(6), base position(3), ...].
 */
class PelvisHeightConstraint final : public StateConstraint {
 public:
  explicit PelvisHeightConstraint(scalar_t minimumHeight)
      : StateConstraint(ConstraintOrder::Linear), minimumHeight_(minimumHeight) {
    if (!std::isfinite(minimumHeight) || minimumHeight <= 0.0) {
      throw std::invalid_argument("pelvis_height_constraint.minimumHeight must be finite and positive");
    }
  }

  PelvisHeightConstraint* clone() const override { return new PelvisHeightConstraint(*this); }
  size_t getNumConstraints(scalar_t /*time*/) const override { return 1; }

  vector_t getValue(scalar_t /*time*/, const vector_t& state, const PreComputation& /*preComp*/) const override {
    return vector_t::Constant(1, state(8) - minimumHeight_);
  }

  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time, const vector_t& state,
                                                          const PreComputation& preComp) const override {
    auto approximation = VectorFunctionLinearApproximation::Zero(1, state.size(), 0);
    approximation.f = getValue(time, state, preComp);
    approximation.dfdx(0, 8) = 1.0;
    return approximation;
  }

 private:
  scalar_t minimumHeight_;
};

}  // namespace ocs2::humanoid
