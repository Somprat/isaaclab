#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <ocs2_core/cost/StateCost.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include <humanoid_common_mpc/common/MpcRobotModelBase.h>

namespace ocs2::humanoid {

// 0.5 * weight * (z_frame(q) - target)^2 on a frame origin height, active at all times.
// Gauss-Newton Hessian from the analytic frame Jacobian; the base joint is translation + ZYX Euler, so v = dq.
class FrameHeightCost final : public StateCost {
 public:
  FrameHeightCost(const PinocchioInterface& pinocchioInterface, const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                  const std::string& frameName, scalar_t target, scalar_t weight)
      : pinocchioInterface_(pinocchioInterface), mpcRobotModel_(&mpcRobotModel), target_(target), weight_(weight) {
    const auto& model = pinocchioInterface_.getModel();
    frameId_ = model.getFrameId(frameName);
    if (frameId_ >= static_cast<pinocchio::FrameIndex>(model.nframes)) throw std::runtime_error("Frame not in MPC model: " + frameName);
    if (model.nq != model.nv) throw std::runtime_error("FrameHeightCost requires nq == nv.");
  }
  FrameHeightCost* clone() const override { return new FrameHeightCost(*this); }

  scalar_t getValue(scalar_t, const vector_t& state, const TargetTrajectories&, const PreComputation&) const override {
    const scalar_t error = height(state) - target_;
    return 0.5 * weight_ * error * error;
  }

  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t, const vector_t& state, const TargetTrajectories&,
                                                                 const PreComputation&) const override {
    const auto& model = pinocchioInterface_.getModel();
    auto& data = pinocchioInterface_.getData();
    const vector_t q = mpcRobotModel_->getGeneralizedCoordinates(state);
    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    matrix_t jacobian = matrix_t::Zero(6, model.nv);
    pinocchio::getFrameJacobian(model, data, frameId_, pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
    const vector_t dz = jacobian.row(2).transpose();
    const scalar_t error = data.oMf[frameId_].translation().z() - target_;
    ScalarFunctionQuadraticApproximation approximation(state.size(), 0);
    approximation.f = 0.5 * weight_ * error * error;
    approximation.dfdx.setZero();
    approximation.dfdx.tail(q.size()) = weight_ * error * dz;
    approximation.dfdxx.setZero();
    approximation.dfdxx.bottomRightCorner(q.size(), q.size()) = weight_ * dz * dz.transpose();
    return approximation;
  }

  scalar_t height(const vector_t& state) const {
    const auto& model = pinocchioInterface_.getModel();
    auto& data = pinocchioInterface_.getData();
    pinocchio::forwardKinematics(model, data, mpcRobotModel_->getGeneralizedCoordinates(state));
    return pinocchio::updateFramePlacement(model, data, frameId_).translation().z();
  }

 private:
  FrameHeightCost(const FrameHeightCost& other) = default;
  mutable PinocchioInterface pinocchioInterface_;
  const MpcRobotModelBase<scalar_t>* mpcRobotModel_;
  pinocchio::FrameIndex frameId_;
  scalar_t target_, weight_;
};

}  // namespace ocs2::humanoid
