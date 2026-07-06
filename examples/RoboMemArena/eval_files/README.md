# RoboMemArena StarVLA Eval

Start a StarVLA TwoChunk server with a RoboMemArena checkpoint:

```bash
CKPT=/path/to/checkpoints/steps_xxx_pytorch_model.pt \
GPU_ID=0 PORT=6697 \
bash examples/RoboMemArena/eval_files/run_twochunk_policy_server.sh
```

Run evaluation from another shell:

```bash
CKPT=/path/to/checkpoints/steps_xxx_pytorch_model.pt \
SUBSET=all NUM_TRIALS_PER_TASK=50 PORT=6697 \
bash examples/RoboMemArena/eval_files/eval_robomemarena_twochunk.sh
```

`SUBSET` can be `all`, `sequence`, `counting`, `transferring`, or `occlusion`.
Use `TASK_IDS=1,2,6-10` to override the subset.

Keep `REPLAN_STEPS` equal to the checkpoint/server `vision_refresh_steps`.
For the current TwoChunk configs this is `8`. DCTMemory is selected by the
checkpoint YAML and server metadata; the eval script does not hard-code it.
