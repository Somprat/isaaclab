# Isolated training environment

This repository uses its own `.venv` and an independent Isaac Lab source checkout
at `.deps/IsaacLab` (v2.2.0). It does not use `../IsaacLab/.venv`.

The selected stack follows Isaac Lab 2.2 installation guidance:
https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/pip_installation.html

- Python 3.11
- Isaac Lab v2.2.0
- Isaac Sim 5.0.0
- PyTorch 2.7.0 / torchvision 0.22.0, CUDA 12.8 wheels
- RSL-RL 2.3.3

Use the interpreter explicitly to avoid accidentally selecting another environment:

```bash
cd /home/choon/isaaclab/general-lowbody
.venv/bin/python scripts/train.py --task G1-Dual-Arm-IK \
  --num_envs 8 --max_iterations 1 --headless --device cuda:0
```

The robot asset is installed at
`source/general_low_body/general_low_body/assets/data/g1.usd`, with supporting
layers under `data/configuration/`. It is a copy of the local G1 29-DOF revision
1.0 asset; the original walking setup was not modified.

cuRobo imports are optional for the DLS task. Selecting a cuRobo task still
requires installing cuRobo. The recorder substep hook is guarded for Isaac Lab
2.2 compatibility. `tensordict==0.8.3` is installed in this environment.

A one-iteration smoke test passed with 8 environments (192 total timesteps).
Checkpoint: `logs/rsl_rl/dual_arm_ik/2026-11-09_00-48-55/model_0.pt`.
This validates startup, rollout, PPO update, and checkpoint saving; it is not a
trained policy or a test of long-run stability.

Your existing walking and hand-training commands continue to run from
`/home/choon/isaaclab/IsaacLab` as before. If you activate this environment with
`source .venv/bin/activate`, run `deactivate` before switching workflows.

## Validation

`uv pip check --python .venv/bin/python` reports a consistent package set.
CUDA is available to PyTorch, and the interpreter paths do not include the old
Isaac Lab checkout. `requirements-local.lock.txt` records installed versions
and local editable source locations (it is specific to this checkout).

The repository PPO import was adjusted to the RSL-RL 2.3.3 location:
`rsl_rl.modules.RandomNetworkDistillation`.

## External hand targets during inference

`play.py` accepts absolute world XYZ positions (meters) independently for each hand:

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python scripts/play.py \
  --task G1-Dual-Arm-IK --num_envs 1 --headless --device cuda:0 \
  --checkpoint /absolute/path/to/model.pt \
  --left-hand-position-world 0.40 0.18 1.00 \
  --right-hand-position-world 0.40 -0.18 1.00 \
  --max-steps 1000
```

These optional flags call `ImpedanceCommand.set_target_world(position, env_ids=None)`
before initial observations. One XYZ vector broadcasts to the selected environments;
a batch can supply one XYZ vector per selected environment. New values must be finite.
Without these flags, the existing fixed-reach sampling behavior remains active.

The setter marks the selected environments as externally controlled. Timer resampling
and episode resets preserve their desired position and orientation. Reference buffers
still reset to the measured hand pose on episode reset, then evolve with the impedance
model. Force and impedance-parameter resampling continue as before. Position input
retains the current sampled desired orientation.

For runtime updates, call the setter when new input arrives, before the normal command
and observation update, so the next policy observation and IK reference share the same
goal. Do not overwrite `ref_pos_w` directly: it is the integrated reference state.
