#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>
#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>
#include <humanoid_common_mpc/swing_foot_planner/FootReferencePlan.h>
#include <ocs2_core/reference/ModeSchedule.h>

namespace ocs2::humanoid {

// Alternating-touchdown displacement and interval (including double support).
struct StepCommand {
  double length;
  double duration;
};

struct PlannedStep {
  double start, liftOff, touchDown;
  size_t foot;
  StepCommand command;
  vector3_t from, to;
};

// An immutable, world-anchored plan. Commands latch at the preceding touchdown;
// subsequent MPC horizons query the same plan rather than moving its anchors.
// Flat ground, fixed heading, two alternating feet. Z remains with the swing planner.
class StepReferencePlan final : public FootReferencePlan {
 public:
  StepReferencePlan(const std::array<vector3_t, 2>& initialFeet, double heading, double finalTime,
                    double doubleSupportFraction, const std::function<StepCommand(double)>& command,
                    double positionWeight = 1000.0)
      : initialFeet_(initialFeet), positionWeight_(positionWeight), heading_(heading) {
    if (!std::isfinite(finalTime) || finalTime <= 0 || !std::isfinite(heading) ||
        !std::isfinite(doubleSupportFraction) || doubleSupportFraction <= 0 || doubleSupportFraction >= 1 ||
        !std::isfinite(positionWeight) || positionWeight <= 0 ||
        !initialFeet[0].allFinite() || !initialFeet[1].allFinite()) {
      throw std::invalid_argument("Invalid step plan geometry, horizon, double support fraction, or tracking weight.");
    }
    const vector3_t forward(std::cos(heading), std::sin(heading), 0.0);
    auto anchors = initialFeet;
    modes_ = ModeSchedule({}, {STANCE});
    double start = 0.0;
    size_t foot = 1;  // First swing is right; left is support.
    while (start < finalTime) {
      const auto c = command(start);
      if (!std::isfinite(c.length) || c.length < 0 || !std::isfinite(c.duration) || c.duration < 0.02) {
        throw std::invalid_argument("Step length must be nonnegative and duration at least 0.02 seconds.");
      }
      // A zero-length startup command is a settling interval in double support,
      // not a full-height step in place. Do not consume the first swing foot.
      if (c.length <= 1e-9) {
        steps_.push_back({start, start + c.duration, start + c.duration, foot, c, anchors[foot], anchors[foot]});
        start += c.duration;
        continue;
      }
      auto to = anchors[foot];
      to += forward * (forward.dot(anchors[1 - foot] - to) + c.length);
      const double lift = start + doubleSupportFraction * c.duration;
      const double touch = start + c.duration;
      steps_.push_back({start, lift, touch, foot, c, anchors[foot], to});
      modes_.eventTimes.push_back(lift);
      modes_.modeSequence.push_back(foot == 1 ? LF : RF);
      modes_.eventTimes.push_back(touch);
      modes_.modeSequence.push_back(STANCE);
      anchors[foot] = to;
      start = touch;
      foot = 1 - foot;
    }
  }

  const ModeSchedule& modeSchedule() const override { return modes_; }
  const std::vector<PlannedStep>& steps() const { return steps_; }
  double positionWeight() const override { return positionWeight_; }
  double heading() const override { return heading_; }

  const PlannedStep& stepAt(double time) const {
    auto it = std::upper_bound(steps_.begin(), steps_.end(), time,
                              [](double t, const PlannedStep& s) { return t < s.start; });
    return it == steps_.begin() ? steps_.front() : *std::prev(it);
  }

  // Normalized vertical-load reference. Unload the next swing foot during
  // existing double support; no contact-event times or step commands change.
  std::array<double, 2> supportWeights(double time) const override {
    std::array<double, 2> weights{0.5, 0.5};
    for (const auto& step : steps_) {
      if (step.command.length <= 1e-9) continue;
      if (time < step.start) return weights;
      std::array<double, 2> target{0.0, 0.0};
      target[1 - step.foot] = 1.0;
      if (time < step.liftOff) {
        const double a = std::clamp((time-step.start)/(step.liftOff-step.start),0.0,1.0);
        const double blend = a*a*a*(10.0+a*(-15.0+6.0*a));
        for (size_t foot=0;foot<2;++foot) weights[foot] += blend*(target[foot]-weights[foot]);
        return weights;
      }
      weights = target;
      if (time < step.touchDown) return weights;
    }
    return weights;
  }

  HorizontalFootReference sample(size_t foot, double time) const override {
    if (foot >= 2) throw std::out_of_range("Step plan supports two feet.");
    vector3_t position = initialFeet_[foot];
    for (const auto& s : steps_) {
      if (s.foot != foot) continue;
      if (time <= s.liftOff) break;
      if (time >= s.touchDown) { position = s.to; continue; }
      const double duration = s.touchDown - s.liftOff;
      const double a = (time - s.liftOff) / duration;
      const double blend = a * a * a * (10.0 + a * (-15.0 + 6.0 * a));
      const double rate = 30.0 * a * a * (1.0 - a) * (1.0 - a) / duration;
      return {s.from + blend * (s.to - s.from), rate * (s.to - s.from), 1.0 - 12.0 * a * a * (1.0 - a) * (1.0 - a)};
    }
    return {position, vector3_t::Zero(), 1.0};
  }

 private:
  std::array<vector3_t, 2> initialFeet_;
  double positionWeight_;
  double heading_;
  std::vector<PlannedStep> steps_;
  ModeSchedule modes_;
};
}  // namespace ocs2::humanoid
