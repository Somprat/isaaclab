#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <utility>

namespace ocs2::humanoid {

// Stride is forward displacement over a full left/right cycle, in meters.
struct MotionCondition {
  double velocity;
  double strideLength;
};

struct ConditionSample {
  MotionCondition command;
  MotionCondition target;
  size_t stage;
  bool transition;
};

// Integer frame allocation keeps exported command labels within one frame of
// the requested ratio and balances plateau frames across the supplied pairs.
class MotionConditioning {
 public:
  MotionConditioning(std::vector<MotionCondition> conditions, size_t frames, double fps, double plateauRatio)
      : conditions_(std::move(conditions)), frames_(frames), fps_(fps) {
    if (conditions_.empty() || !std::isfinite(fps) || fps <= 0 || !std::isfinite(plateauRatio) ||
        plateauRatio <= 0 || plateauRatio >= 1 || frames == 0) {
      throw std::invalid_argument("Conditioning needs pairs, positive fps/frames, and plateau ratio in (0,1).");
    }
    for (const auto& c : conditions_) {
      if (!std::isfinite(c.velocity) || !std::isfinite(c.strideLength) || c.velocity <= 0 || c.strideLength <= 0) {
        throw std::invalid_argument("Condition velocities and full stride lengths must be finite and positive.");
      }
    }
    const size_t plateaus = static_cast<size_t>(std::llround(frames * plateauRatio));
    const size_t transitions = frames - plateaus;
    if (plateaus < conditions_.size() || transitions < 2 * conditions_.size()) {
      throw std::invalid_argument("Duration is too short: each pair needs a plateau frame and at least two transition frames.");
    }
    size_t start = 0;
    for (size_t i = 0; i < conditions_.size(); ++i) {
      const size_t ramp = transitions / conditions_.size() + (i < transitions % conditions_.size());
      const size_t hold = plateaus / conditions_.size() + (i < plateaus % conditions_.size());
      starts_.push_back(start);
      ramps_.push_back(ramp);
      start += ramp + hold;
    }
  }

  ConditionSample sample(double time) const {
    double frame = std::clamp(time * fps_, 0.0, static_cast<double>(frames_ - 1));
    if (std::abs(frame - std::round(frame)) < 1e-8) frame = std::round(frame);
    const auto upper = std::upper_bound(starts_.begin(), starts_.end(), frame);
    const size_t stage = static_cast<size_t>(upper - starts_.begin() - 1);
    const double offset = frame - starts_[stage];
    const auto target = conditions_[stage];
    const bool transition = offset < ramps_[stage];
    const auto source = stage == 0 ? MotionCondition{0.0, target.strideLength} : conditions_[stage - 1];
    const double a = std::clamp(offset / static_cast<double>(ramps_[stage] - 1), 0.0, 1.0);
    const double s = a * a * a * (10.0 + a * (-15.0 + 6.0 * a));
    return {{source.velocity + s * (target.velocity - source.velocity),
             source.strideLength + s * (target.strideLength - source.strideLength)}, target, stage, transition};
  }

 private:
  std::vector<MotionCondition> conditions_;
  size_t frames_;
  double fps_;
  std::vector<size_t> starts_;
  std::vector<size_t> ramps_;
};

}  // namespace ocs2::humanoid
