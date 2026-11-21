#include <humanoid_centroidal_mpc_ros2/MotionConditioning.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using ocs2::humanoid::MotionCondition;
using ocs2::humanoid::MotionConditioning;
void require(bool value) { if (!value) throw std::runtime_error("Conditioning assertion failed"); }
template <typename F> void rejects(F f) {
  bool rejected = false;
  try { f(); } catch (const std::invalid_argument&) { rejected = true; }
  require(rejected);
}
int main() {
  const std::vector<MotionCondition> pairs{{0.3, 0.36}, {0.7, 0.49}, {0.4, 0.52}};
  MotionConditioning schedule(pairs, 601, 30.0, 0.8);
  size_t transitions = 0, holds[3]{};
  for (size_t frame = 0; frame < 601; ++frame) {
    const auto c = schedule.sample(frame / 30.0);
    require(c.command.strideLength > 0 && c.command.velocity >= 0);
    require(std::isfinite(c.command.velocity / c.command.strideLength));
    if (c.transition) {
      ++transitions;
      const auto from = c.stage == 0 ? MotionCondition{0.0, pairs[0].strideLength} : pairs[c.stage - 1];
      require(c.command.velocity >= std::min(from.velocity, c.target.velocity) - 1e-12);
      require(c.command.velocity <= std::max(from.velocity, c.target.velocity) + 1e-12);
    } else {
      ++holds[c.stage];
      require(std::abs(c.command.velocity - c.target.velocity) < 1e-12);
      require(std::abs(c.command.strideLength - c.target.strideLength) < 1e-12);
    }
  }
  require(transitions == 120 && holds[0] == 161 && holds[1] == 160 && holds[2] == 160);
  require(schedule.sample(0).command.velocity == 0);
  require(schedule.sample(1000).command.velocity == pairs.back().velocity);
  for (double t : {39.0 / 30, 201.0 / 30, 240.0 / 30}) {
    const auto l = schedule.sample(t - 1e-5).command, r = schedule.sample(t + 1e-5).command;
    require(std::abs(r.velocity - l.velocity) < 1e-7);
    require(std::abs(r.strideLength - l.strideLength) < 1e-7);
  }
  MotionConditioning single({{0.63, 0.42}}, 151, 30, 0.6);
  require(single.sample(5).target.velocity == 0.63);
  rejects([] { MotionConditioning s({}, 100, 30, 0.8); });
  rejects([] { MotionConditioning s({{0, 0.4}}, 100, 30, 0.8); });
  rejects([] { MotionConditioning s({{0.4, -1}}, 100, 30, 0.8); });
  rejects([] { MotionConditioning s({{std::numeric_limits<double>::infinity(), 1}}, 100, 30, 0.8); });
  rejects([] { MotionConditioning s({{0.4, 0.4}}, 100, 30, 1.0); });
  rejects([&] { MotionConditioning s(pairs, 3, 30, 0.8); });
  std::cout << "Condition schedule tests passed\n";
}
