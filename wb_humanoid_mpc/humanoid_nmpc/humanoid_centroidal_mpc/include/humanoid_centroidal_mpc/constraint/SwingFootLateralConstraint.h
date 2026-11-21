#pragma once

#include <humanoid_common_mpc/constraint/EndEffectorKinematicsLinearVelConstraint.h>
#include <humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h>

namespace ocs2::humanoid {

/** Track only the lateral swing coordinate; forward placement remains step-conditioned. */
class SwingFootLateralConstraint final : public StateInputConstraint {
 public:
  SwingFootLateralConstraint(const SwitchedModelReferenceManager& manager,
                            const EndEffectorKinematics<scalar_t>& kinematics, size_t foot)
      : StateInputConstraint(ConstraintOrder::Linear), manager_(&manager), foot_(foot),
        constraint_(std::make_unique<EndEffectorKinematicsLinearVelConstraint>(kinematics, 1)) {}
  SwingFootLateralConstraint(const SwingFootLateralConstraint& other)
      : StateInputConstraint(other), manager_(other.manager_), foot_(other.foot_), constraint_(other.constraint_->clone()) {}
  SwingFootLateralConstraint* clone() const override { return new SwingFootLateralConstraint(*this); }
  bool isActive(scalar_t time) const override {
    return manager_->getStepReferencePlan() && !manager_->getContactFlags(time)[foot_];
  }
  size_t getNumConstraints(scalar_t /*time*/) const override { return 1; }
  vector_t getValue(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) const override {
    configure(time);
    return constraint_->getValue(time, state, input, preComp);
  }
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time, const vector_t& state, const vector_t& input,
                                                           const PreComputation& preComp) const override {
    configure(time);
    return constraint_->getLinearApproximation(time, state, input, preComp);
  }
 private:
  void configure(scalar_t time) const {
    const auto& plan = manager_->getStepReferencePlan();
    const auto reference = plan->sample(foot_, time);
    const vector3_t lateral(-std::sin(plan->heading()), std::cos(plan->heading()), 0.0);
    EndEffectorKinematicsLinearVelConstraint::Config config;
    config.Av = lateral.transpose();
    config.Ax = 5.0 * config.Av;
    config.b = vector_t::Constant(1, -lateral.dot(reference.velocity + 5.0 * reference.position));
    constraint_->configure(std::move(config));
  }
  const SwitchedModelReferenceManager* manager_;
  size_t foot_;
  std::unique_ptr<EndEffectorKinematicsLinearVelConstraint> constraint_;
};

}  // namespace ocs2::humanoid
