#include <humanoid_centroidal_mpc_ros2/StepConditioning.h>
#include <humanoid_common_mpc/gait/GaitSchedule.h>
#include <humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h>
#include <iostream>
#include <limits>
using namespace ocs2::humanoid;
#define check(ok) do { if (!(ok)) throw std::runtime_error("Step reference assertion failed at line " + std::to_string(__LINE__)); } while (false)
void near(double a, double b, double tol = 1e-10) {
  if (!(std::abs(a-b) < tol)) throw std::runtime_error("Expected " + std::to_string(a) + " near " + std::to_string(b));
}
template<class F> void rejects(F f) {
  bool rejected = false;
  try { f(); } catch (const std::invalid_argument&) { rejected = true; }
  check(rejected);
}
int main() {
  const std::array<ocs2::vector3_t, 2> feet{ocs2::vector3_t(0, .1, 0), ocs2::vector3_t(0, -.1, 0)};
  // Same length, different durations: geometry unchanged, timing independently scales.
  StepReferencePlan slow(feet, 0, 2, .2, [](double) { return StepCommand{.2, .5}; });
  StepReferencePlan fast(feet, 0, 1, .2, [](double) { return StepCommand{.2, .25}; });
  StepReferencePlan longer(feet, 0, 2, .2, [](double) { return StepCommand{.3, .5}; });
  near(slow.steps().size(), 4);
  for (size_t i=0; i<4; ++i) {
    const auto& s=slow.steps()[i];
    near(s.to.x(), (i+1)*.2);
    near(s.to.y(), s.foot == 1 ? -.1 : .1);
    check(s.foot == (i%2 == 0 ? 1 : 0));
    near(s.touchDown-s.start, .5);
    near(fast.steps()[i].to.x(), s.to.x());
    near(fast.steps()[i].touchDown*2, s.touchDown);
    near(longer.steps()[i].touchDown, s.touchDown);
    near(longer.steps()[i].to.x(), (i+1)*.3);
    near(slow.sample(s.foot,s.liftOff).position.x(), s.from.x());
    near(slow.sample(s.foot,s.touchDown).position.x(), s.to.x());
    near(slow.sample(s.foot,s.touchDown).velocity.norm(),0);
    for (double t=s.start; t<s.touchDown; t+=.007) {
      const auto ref=slow.sample(s.foot,t);
      check(ref.position.x()>=s.from.x()-1e-10 && ref.position.x()<=s.to.x()+1e-10);
      near(slow.sample(1-s.foot,t).velocity.norm(),0);
      const ocs2::vector3_t finiteDiff=(slow.sample(s.foot,t+1e-6).position-slow.sample(s.foot,t-1e-6).position)/2e-6;
      check((finiteDiff-ref.velocity).norm()<1e-7);
    }
    check(slow.modeSchedule().modeAtTime((s.liftOff+s.touchDown)/2)==(s.foot==1?LF:RF));
    check(slow.modeSchedule().modeAtTime((s.start+s.liftOff)/2)==STANCE);
  }
  // C2 arch: zero endpoint velocity/acceleration, prescribed peak, analytic derivatives.
  SplineCpg smoothArch({.2,0,0},.08,{1.0,-.001,0},true);
  near(smoothArch.position(.2),0);
  near(smoothArch.position(1.0),-.001);
  near(smoothArch.position(.6),.08);
  for (double t : {.2,1.0}) {
    near(smoothArch.velocity(t),0);
    near(smoothArch.acceleration(t),0);
  }
  for (double t=.21;t<1.0;t+=.013) {
    near((smoothArch.position(t+1e-6)-smoothArch.position(t-1e-6))/2e-6,smoothArch.velocity(t),1e-7);
    near((smoothArch.velocity(t+1e-6)-smoothArch.velocity(t-1e-6))/2e-6,smoothArch.acceleration(t),1e-7);
  }
  const auto& first=slow.steps().front();
  near(slow.sample(first.foot,first.liftOff).weightScale,1);
  near(slow.sample(first.foot,first.liftOff+1e-8).weightScale,1);
  near(slow.sample(first.foot,first.touchDown-1e-8).weightScale,1);
  near(slow.sample(first.foot,(first.liftOff+first.touchDown)/2).weightScale,.25);
  // Startup rests in double support; the first actual swing remains right.
  StepReferencePlan startup(feet,0,5,.2,[](double t){return StepCommand{t<.8?0.0:.1,.8};});
  near(startup.steps()[0].liftOff,startup.steps()[0].touchDown);
  check(startup.steps()[1].foot==1);
  for (double t=0;t<.8;t+=.01) {
    check(startup.modeSchedule().modeAtTime(t)==STANCE);
    for (size_t foot=0;foot<2;++foot) {
      near((startup.sample(foot,t).position-feet[foot]).norm(),0);
      near(startup.sample(foot,t).velocity.norm(),0);
    }
  }
  check(startup.modeSchedule().modeAtTime(1.1)==LF);
  near(startup.supportWeights(.4)[0],.5);
  near(startup.supportWeights(.8)[0],.5);
  const auto& firstSwing = startup.steps()[1];
  near(startup.supportWeights(firstSwing.liftOff)[0],1.0);
  near(startup.supportWeights(firstSwing.liftOff)[1],0.0);
  for (double t=0;t<4.5;t+=.003) {
    const auto weights=startup.supportWeights(t);
    near(weights[0]+weights[1],1.0);
    const auto contacts=modeNumber2StanceLeg(startup.modeSchedule().modeAtTime(t));
    for (size_t foot=0;foot<2;++foot) {
      check(weights[foot]>=0 && weights[foot]<=1);
      if (!contacts[foot]) near(weights[foot],0);
      check(std::abs(startup.supportWeights(t+1e-7)[foot]-weights[foot])<1e-5);
    }
  }

  GaitSchedule startupGait(startup.modeSchedule(),ModeSequenceTemplate({0.0,1.0},{STANCE}),0.0);
  SwingTrajectoryPlanner startupVertical(SwingTrajectoryPlanner::Config{},2);
  startupVertical.update(startupGait.getModeSchedule(-1,3),0.0);
  for (double t=0;t<.8;t+=.01) {
    for (size_t foot=0;foot<2;++foot) near(startupVertical.getZpositionConstraint(foot,t),0);
  }
  // Existing scheduling and vertical swing code must preserve the supplied
  // contacts through repeated moving-horizon updates.
  StepReferencePlan longPlan(feet,0,10,.2,[](double){return StepCommand{.2,.5};});
  GaitSchedule gait(longPlan.modeSchedule(), ModeSequenceTemplate({0.0,1.0},{STANCE}),0.0);
  SwingTrajectoryPlanner vertical(SwingTrajectoryPlanner::Config{},2);
  for (double t=0;t<5;t+=.033) {
    const auto modes=gait.getModeSchedule(t-1,t+2);
    vertical.update(modes,0.0);
    for (double horizon=t;horizon<t+1;horizon+=.071) {
      check(modes.modeAtTime(horizon)==longPlan.modeSchedule().modeAtTime(horizon));
      const auto contact=modeNumber2StanceLeg(modes.modeAtTime(horizon));
      for (size_t foot=0;foot<2;++foot) {
        check(std::isfinite(vertical.getZpositionConstraint(foot,horizon)));
        if (contact[foot]) {
          near(vertical.getZpositionConstraint(foot,horizon),0);
          near(longPlan.sample(foot,horizon).velocity.norm(),0);
        } else {
          check(vertical.getZpositionConstraint(foot,horizon)>=-1e-10);
        }
      }
    }
  }
  // Rotated/translated initial state: distances along heading and lateral offsets preserved.
  const double yaw=.7;
  const ocs2::vector3_t forward(std::cos(yaw),std::sin(yaw),0), lateral(-std::sin(yaw),std::cos(yaw),0), origin(4,-3,.02);
  StepReferencePlan rotated({origin+.1*lateral,origin-.1*lateral},yaw,2,.2,[](double){return StepCommand{.2,.5};});
  near(rotated.heading(),yaw);
  for (size_t i=0;i<4;++i) {
    const auto& s=rotated.steps()[i];
    near(forward.dot(s.to-origin),(i+1)*.2);
    near(lateral.dot(s.to-origin),s.foot==1?-.1:.1);
  }
  // A mid-step command change takes effect only at the next boundary.
  StepReferencePlan transition(feet,0,2,.2,[](double t){return t<.2?StepCommand{.2,.5}:StepCommand{.3,.4};});
  near(transition.stepAt(.3).command.length,.2);
  near(transition.stepAt(.5).command.length,.3);
  near(transition.steps()[1].to.x(),.5);
  near(transition.steps()[1].touchDown,.9);
  StepConditioning schedule({{.1,.5},{.2,.5},{.2,.3}},601,30,.8);
  near(schedule.sample(0).command.length,0);
  near(schedule.sample(0).command.duration,.5);
  near(schedule.sample(100).command.length,.2);
  near(schedule.sample(100).command.duration,.3);
  // Direct mode ignores plateau allocation: settle once, then full commands.
  StepConditioning direct({{.1,.8}},601,30,.8,false);
  near(direct.sample(0).command.length,0);
  near(direct.sample(.79).command.length,0);
  near(direct.sample(.8).command.length,.1);
  near(direct.sample(.8).command.duration,.8);
  check(!direct.sample(.8).transition);
  near(direct.sample(19).command.length,.1);
  StepReferencePlan directPlan(feet,0,5,.2,[&](double t){return direct.sample(t).command;});
  near(directPlan.steps()[0].command.length,0);
  near(directPlan.steps()[1].command.length,.1);
  near(directPlan.steps()[1].touchDown,1.6);
  StepConditioning directStages({{.1,.8},{.2,.6}},601,30,.01,false);
  near(directStages.sample(9.99).command.length,.1);
  near(directStages.sample(10.0).command.length,.2);
  near(directStages.sample(10.0).command.duration,.6);
  near(directStages.sample(100).command.length,.2);
  StepConditioning thirds({{.1,.8},{.2,.8},{.3,.8}},601,30,.8,false);
  near(thirds.sample(6.66).command.length,.1);
  near(thirds.sample(20.0/3).command.length,.2);
  near(thirds.sample(13.33).command.length,.2);
  near(thirds.sample(40.0/3).command.length,.3);
  StepReferencePlan thirdPlan(feet,0,20,.2,[&](double t){return thirds.sample(t).command;});
  near(thirdPlan.stepAt(6.8).command.length,.1);
  near(thirdPlan.stepAt(7.21).command.length,.2);
  near(thirdPlan.stepAt(13.61).command.length,.3);

  rejects([&]{StepConditioning s({},601,30,.8,false);});
  rejects([&]{StepConditioning s({{.1,.8}},1,30,.8,false);});
  rejects([&]{StepConditioning s({{.1,.8}},10,30,.8,false);});
  rejects([&]{StepConditioning s({{0,.8}},601,30,.8,false);});
  rejects([&]{StepReferencePlan p(feet,0,1,0,[](double){return StepCommand{.2,.5};});});
  rejects([&]{StepReferencePlan p(feet,0,1,.2,[](double){return StepCommand{.2,0};});});
  rejects([&]{StepReferencePlan p(feet,0,1,.2,[](double){return StepCommand{-1,.5};});});
  rejects([&]{StepConditioning s({{.2,std::numeric_limits<double>::quiet_NaN()}},601,30,.8);});
  std::cout << "Step reference geometry, timing, interpolation, latching and validation passed\n";
}
