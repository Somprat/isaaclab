// Small checks for the sample-backed LIP foot plan: contact timing, between-sample queries, vertical override.
#include <humanoid_common_mpc/gait/GaitSchedule.h>
#include <humanoid_common_mpc/swing_foot_planner/LipReferencePlan.h>
#include <humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace ocs2::humanoid;
using ocs2::vector3_t;
#define check(ok) do { if (!(ok)) throw std::runtime_error("LIP plan assertion failed at line " + std::to_string(__LINE__)); } while (false)
void near(double a, double b, double tol = 1e-10) {
  if (!(std::abs(a - b) < tol)) throw std::runtime_error("Expected " + std::to_string(a) + " near " + std::to_string(b));
}

int main() {
  // 0.01 s samples: double support until 0.1, right swing until 0.3, then left swing until 0.5 (no double support).
  // Right foot moves along x = 0.2 * smoothstep over its swing with z = 0.04 sin(pi s); CoM x = 0.5 t, constant height.
  std::vector<LipSample> samples;
  for (int k = 0; k <= 50; ++k) {
    const double t = 0.01 * k;
    LipSample s{t, {true, true}, vector3_t(0.5 * t, 0, 0.69), vector3_t(0.5, 0, 0), {}, {}};
    s.footPosition = {vector3_t(0, 0.1, 0), vector3_t(0, -0.1, 0)};
    s.footVelocity = {vector3_t::Zero(), vector3_t::Zero()};
    if (t >= 0.1 - 1e-9 && t < 0.3 - 1e-9) {
      s.contact = {true, false};
      const double a = (t - 0.1) / 0.2;
      s.footPosition[1] = vector3_t(0.2 * a * a * (3 - 2 * a), -0.1, 0.04 * std::sin(M_PI * a));
      s.footVelocity[1] = vector3_t(0.2 * 6 * a * (1 - a) / 0.2, 0, 0.04 * M_PI * std::cos(M_PI * a) / 0.2);
    } else if (t >= 0.3 - 1e-9) {
      s.contact = {false, true};
      s.footPosition[1] = vector3_t(0.2, -0.1, 0);
    }
    samples.push_back(s);
  }
  const LipReferencePlan plan(samples, 0.0, 500.0);

  // Support mask -> mode schedule; each change takes effect at the first sample showing it.
  const auto& modes = plan.modeSchedule();
  check(modes.modeSequence.size() == 4);
  check(modes.modeSequence[0] == STANCE && modes.modeSequence[1] == LF && modes.modeSequence[2] == RF && modes.modeSequence[3] == STANCE);
  near(modes.eventTimes[0], 0.1, 1e-9);
  near(modes.eventTimes[1], 0.3, 1e-9);
  near(modes.eventTimes[2], 0.5, 1e-9);
  check(modes.modeAtTime(0.0999) == STANCE && modes.modeAtTime(0.1001) == LF && modes.modeAtTime(0.31) == RF);
  check(plan.contactFlags(0.2)[0] && !plan.contactFlags(0.2)[1]);
  near(plan.supportWeights(0.05)[0], 0.5);
  near(plan.supportWeights(0.2)[0], 1.0);
  near(plan.supportWeights(0.2)[1], 0.0);

  // Between samples: Hermite position uses sample velocities; exact at samples, continuous in between.
  near(plan.sample(1, 0.2).position.x(), 0.1, 1e-12);
  const double tq = 0.2037;  // not a sample time
  const double a = (tq - 0.1) / 0.2;
  near(plan.sample(1, tq).position.x(), 0.2 * a * a * (3 - 2 * a), 1e-6);
  near(plan.sample(1, tq).position.z(), 0.04 * std::sin(M_PI * a), 1e-5);
  near(plan.sample(1, tq).velocity.x(), 6 * a * (1 - a), 5e-3);  // linear velocity interpolation: h^2 |v''| / 8 = 3.75e-3 here
  near(plan.com(0.12345).first.x(), 0.5 * 0.12345, 1e-12);
  near(plan.com(0.12345).second.x(), 0.5, 1e-12);
  // Outside the samples: hold the end positions at zero velocity.
  near(plan.sample(1, 9.0).position.x(), 0.2);
  near(plan.sample(1, 9.0).velocity.norm(), 0.0);
  near(plan.positionWeight(), 500.0);

  // Swing planner: the vertical reference replaces generated height splines over a moving horizon.
  GaitSchedule gait(plan.modeSchedule(), ModeSequenceTemplate({0.0, 1.0}, {STANCE}), 0.0);
  SwingTrajectoryPlanner vertical(SwingTrajectoryPlanner::Config{}, 2);
  vertical.update(gait.getModeSchedule(-1, 2), 0.0);
  vertical.setVerticalReference(std::make_shared<LipReferencePlan>(plan));
  for (double t = 0.0; t < 0.5; t += 0.0013) {
    for (size_t foot = 0; foot < 2; ++foot) {
      near(vertical.getZpositionConstraint(foot, t), plan.sample(foot, t).position.z());
      near(vertical.getZvelocityConstraint(foot, t), plan.sample(foot, t).velocity.z());
    }
  }
  vertical.setVerticalReference(nullptr);
  near(vertical.getZpositionConstraint(0, 0.05), 0.0);  // generated spline again: stance foot on the ground

  bool rejected = false;
  try {
    auto flight = samples;
    flight[5].contact = {false, false};
    LipReferencePlan p(flight, 0.0);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected);
  std::cout << "LIP reference plan checks passed\n";
  return 0;
}
