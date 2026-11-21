# Continue fine-tuning the better G1 online-hands policy

From `/home/choon/isaaclab`:

```bash
bash IsaacLab/scripts/reinforcement_learning/finetune_g1_online_hands.sh
```

The launcher loads actor and critic weights directly from
`logs/rsl_rl/g1_lipm_online_hands_finetune/run_v2zonEaZ/training/model_499.pt`.
It uses two GPUs, 2048 environments per GPU, 500 iterations and learning rate
0.0001. The optimizer and iteration counter start fresh. No input-layer transfer
is needed: this checkpoint already has 127 inputs and 29 action outputs.

Hand targets default to `--hand-target-frame world`: offsets use fixed world
X/Y/Z axes with their origin at the current torso position. They translate with
the torso but do not rotate with its heading or tilt. IK converts the offsets to
the current torso frame. Use `--hand-target-frame reset_heading` for the previous
initial-heading axes, or `torso` for offsets that rotate with the torso.
Saved checkpoints retain their configured frame for playback.

The saved reach fraction and effective IK influence are preserved at their
completed stages when using `--from-finetuned`; the curriculum is not restarted.
The trainer applies the current hand endpoints, frame, and reach timing options.
Current training rewards remain unchanged: pelvis/torso heading weights 2.0/1.0,
torso/waist yaw-motion penalties -0.05/-0.02, hand rewards 0.05 each and no nominal
arm-posture penalty. Selecting an earlier model does not revert these objectives.

Every invocation creates a unique `run_XXXXXXXX/training/` folder under
`logs/rsl_rl/g1_lipm_online_hands_finetune/`. The source run is preserved.

Check settings without creating output:

```bash
bash IsaacLab/scripts/reinforcement_learning/finetune_g1_online_hands.sh --dry-run
```

Single-GPU example:

```bash
NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 ENVS_PER_GPU=256 MAX_ITERATIONS=500 \
  bash IsaacLab/scripts/reinforcement_learning/finetune_g1_online_hands.sh
```

The Python trainer accepts `--from-finetuned` for existing online-hands checkpoints.
Without it, the original fresh-transfer path remains supported through
`transfer_g1_online_hands.py`. Production playback uses `play_g1_reference.py` and
restores saved reach/IK stages, recording hand/heading errors and torso/waist
angular velocities alongside an MP4 in a fresh output directory.
