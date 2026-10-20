# MossAD-Demo pilot scripts (MI300X droplet)

Not part of the trainer: the unattended pipeline for the MossAD-Demo MOSS-Image pilot.

- `pipeline.py`: waits for the speed test, picks batch sizes, runs the LR test (A 4e-5, B 2e-5, 10 epochs each at
  512 px), then continues the winner (A unless `/root/work/pipeline/winner` says B) through 512 -> 768 -> 1024.
  Deadline watchdog: `PIPELINE_DEADLINE` (default 2026-10-23T23:00Z), save-and-quit 45 min before.
- `upload_loop.sh`: uploads checkpoints (no optimizer state), samples and configs to `RicemanT/MossAD-Demo`.
- `status.sh`: one-glance progress.
