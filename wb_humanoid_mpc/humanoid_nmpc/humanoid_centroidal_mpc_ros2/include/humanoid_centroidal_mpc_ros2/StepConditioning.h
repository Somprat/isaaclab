#pragma once

#include <optional>

#include <humanoid_centroidal_mpc_ros2/MotionConditioning.h>
#include <humanoid_common_mpc/swing_foot_planner/StepReferencePlan.h>

namespace ocs2::humanoid {
struct StepConditionSample {
  StepCommand command, target;
  size_t stage;
  bool transition;
};

// Reuse the frame allocation and C2 interpolation, but interpolate L and T
// independently, never velocity and length. Start from zero L at finite T.
class StepConditioning {
 public:
  StepConditioning(const std::vector<StepCommand>& conditions, size_t frames, double fps, double plateauRatio,
                   bool interpolate = true)
      : conditions_(conditions) {
    const auto encoded = encode(conditions);
    if (interpolate) {
      schedule_.emplace(encoded, frames, fps, plateauRatio);
    } else {
      if (conditions.empty() || frames < 2 || !std::isfinite(fps) || fps <= 0)
        throw std::invalid_argument("Direct step conditioning needs conditions, at least two frames and positive FPS.");
      settleTime_ = conditions.front().duration;
      const double duration = static_cast<double>(frames - 1) / fps;
      if (duration <= settleTime_)
        throw std::invalid_argument("Motion must be longer than the initial settling interval.");
      stageDuration_ = duration / conditions.size();
    }
  }
  StepConditionSample sample(double time) const {
    if (!schedule_) {
      if (time < settleTime_ - 1e-9)
        return {{0.0, conditions_.front().duration}, conditions_.front(), 0, true};
      const double stage = std::floor(std::max(0.0, time + 1e-9) / stageDuration_);
      const size_t index = static_cast<size_t>(std::min(stage, static_cast<double>(conditions_.size() - 1)));
      return {conditions_[index], conditions_[index], index, false};
    }
    const auto s = schedule_->sample(time);
    return {{s.command.velocity, s.command.strideLength}, {s.target.velocity, s.target.strideLength}, s.stage, s.transition};
  }
 private:
  static std::vector<MotionCondition> encode(const std::vector<StepCommand>& conditions) {
    std::vector<MotionCondition> result;
    for (const auto& c : conditions) {
      if (!std::isfinite(c.length) || c.length <= 0)
        throw std::invalid_argument("Step length must be finite and positive.");
      if (!std::isfinite(c.duration) || c.duration < 0.02)
        throw std::invalid_argument("Step duration must be finite and at least 0.02 seconds.");
      result.push_back({c.length, c.duration});
    }
    return result;
  }
  std::optional<MotionConditioning> schedule_;
  std::vector<StepCommand> conditions_;
  double settleTime_ = 0.0;
  double stageDuration_ = 0.0;
};
}  // namespace ocs2::humanoid
