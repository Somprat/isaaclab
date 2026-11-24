#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>
#include <humanoid_common_mpc/swing_foot_planner/FootReferencePlan.h>

namespace ocs2::humanoid {

// One LIP reference sample (world z-up). Feet are the MPC contact frames, ordered left, right.
struct LipSample {
  double time;
  std::array<bool, 2> contact;
  vector3_t comPosition, comVelocity;
  std::array<vector3_t, 2> footPosition, footVelocity;
};

// Foot plan backed by LIP samples instead of generated footsteps.
// Contact timing comes from the support mask; a contact change takes effect at the first sample that
// shows it. Positions are cubic-Hermite interpolated from sample positions and velocities and
// velocities are linearly interpolated, so any time inside the planning horizon is queried
// continuously, not only at sample times. Queries outside the samples hold the end sample with zero velocity.
class LipReferencePlan final : public FootReferencePlan {
 public:
  LipReferencePlan(std::vector<LipSample> samples, double heading, double positionWeight = 1000.0)
      : samples_(std::move(samples)), heading_(heading), positionWeight_(positionWeight) {
    if (samples_.size() < 2 || !std::isfinite(heading) || !std::isfinite(positionWeight) || positionWeight <= 0) {
      throw std::invalid_argument("LIP plan needs at least two samples, a finite heading, and a positive position weight.");
    }
    for (size_t k = 0; k < samples_.size(); ++k) {
      const auto& s = samples_[k];
      if (!std::isfinite(s.time) || (k > 0 && !(s.time > samples_[k - 1].time))) {
        throw std::invalid_argument("LIP sample times must be finite and strictly increasing.");
      }
      if (!s.contact[0] && !s.contact[1]) {
        throw std::invalid_argument("LIP support mask has a flight sample at t=" + std::to_string(s.time));
      }
      if (!s.comPosition.allFinite() || !s.comVelocity.allFinite()) throw std::invalid_argument("Non-finite LIP CoM sample.");
      for (size_t foot = 0; foot < 2; ++foot) {
        if (!s.footPosition[foot].allFinite() || !s.footVelocity[foot].allFinite()) {
          throw std::invalid_argument("Non-finite LIP foot sample.");
        }
      }
    }
    modes_ = ModeSchedule({}, {modeOf(samples_.front().contact)});
    for (size_t k = 1; k < samples_.size(); ++k) {
      const size_t mode = modeOf(samples_[k].contact);
      if (mode != modes_.modeSequence.back()) {
        modes_.eventTimes.push_back(samples_[k].time);
        modes_.modeSequence.push_back(mode);
      }
    }
    // Past the samples, hold both feet down (GaitSchedule extends the final mode).
    if (modes_.modeSequence.back() != STANCE) {
      modes_.eventTimes.push_back(samples_.back().time);
      modes_.modeSequence.push_back(STANCE);
    }
  }

  const ModeSchedule& modeSchedule() const override { return modes_; }
  double positionWeight() const override { return positionWeight_; }
  double heading() const override { return heading_; }
  bool hasVerticalReference() const override { return true; }

  double startTime() const { return samples_.front().time; }
  double finalTime() const { return samples_.back().time; }
  const std::vector<LipSample>& samples() const { return samples_; }

  std::array<bool, 2> contactFlags(double time) const {
    const size_t k = sampleIndexAtOrBefore(time);
    return samples_[k].contact;
  }

  std::array<double, 2> supportWeights(double time) const override {
    const auto contact = contactFlags(time);
    if (contact[0] && contact[1]) return {0.5, 0.5};
    return {contact[0] ? 1.0 : 0.0, contact[1] ? 1.0 : 0.0};
  }

  HorizontalFootReference sample(size_t foot, double time) const override {
    if (foot >= 2) throw std::out_of_range("LIP plan supports two feet.");
    const auto [position, velocity] =
        interpolate(time, [foot](const LipSample& s) { return std::make_pair(s.footPosition[foot], s.footVelocity[foot]); });
    return {position, velocity, 1.0};
  }

  std::pair<vector3_t, vector3_t> com(double time) const {
    return interpolate(time, [](const LipSample& s) { return std::make_pair(s.comPosition, s.comVelocity); });
  }

 private:
  static size_t modeOf(const std::array<bool, 2>& contact) {
    if (contact[0] && contact[1]) return STANCE;
    return contact[0] ? LF : RF;  // LF/RF name the foot in contact.
  }

  size_t sampleIndexAtOrBefore(double time) const {
    const auto it = std::upper_bound(samples_.begin(), samples_.end(), time,
                                     [](double t, const LipSample& s) { return t < s.time; });
    return it == samples_.begin() ? 0 : static_cast<size_t>(std::distance(samples_.begin(), it) - 1);
  }

  template <typename Getter>
  std::pair<vector3_t, vector3_t> interpolate(double time, Getter get) const {
    if (time <= samples_.front().time) return {get(samples_.front()).first, vector3_t::Zero()};
    if (time >= samples_.back().time) return {get(samples_.back()).first, vector3_t::Zero()};
    const size_t k = sampleIndexAtOrBefore(time);
    const auto& a = samples_[k];
    const auto& b = samples_[k + 1];
    const auto [p0, v0] = get(a);
    const auto [p1, v1] = get(b);
    const double h = b.time - a.time;
    const double s = (time - a.time) / h;
    const double s2 = s * s, s3 = s2 * s;
    const vector3_t position = (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * h * v0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * h * v1;
    return {position, (1.0 - s) * v0 + s * v1};
  }

  std::vector<LipSample> samples_;
  double heading_;
  double positionWeight_;
  ModeSchedule modes_;
};

}  // namespace ocs2::humanoid
