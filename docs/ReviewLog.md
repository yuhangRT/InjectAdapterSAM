# WireCR-HQInstSAM Review Log

## S00

- Status: passed
- Time: 2026-04-02
- Decision: new mainline skeleton created, old mainline frozen

## S01

- Status: passed
- Time: 2026-04-02
- Decision: COCO v2 converter and YAML-first config landed

## S02

- Status: passed
- Time: 2026-04-02
- Decision: dataset, transforms, collate, and visualization smoke path passed

## S03

- Status: passed
- Time: 2026-04-02 17:11:31 UTC
- Decision: SAM backbone v2, LoRA injection, and module registration passed

## S04

- Status: passed
- Time: 2026-04-02 17:20:03 UTC
- Decision: WireCR multiscale adapter and pixel decoder passed

## S05

- Status: passed
- Time: 2026-04-03 00:31:00 UTC
- Decision: query head, matcher, and coarse losses passed

## S06

- Status: passed
- Time: 2026-04-03 00:42:31 UTC
- Decision: prompt builder v2 passed with gt/pred/mixed prompt metadata

## S07

- Status: passed
- Time: 2026-04-03 00:59 UTC
- Decision: HQ refiner, quality head, and refine losses passed

## S08

- Status: passed
- Time: 2026-04-03 00:59 UTC
- Decision: coarse-to-refine closed loop, score fusion, and class-wise mask NMS passed

## S09

- Status: passed
- Time: 2026-04-04 04:34 UTC
- Scope check: unified trainer only, no proposal-only / refine-only / oracle branch
- Test check:
  - `pytest tests/test_trainer_smoke.py -q`
  - `pytest tests/test_backbone_v2.py tests/test_pixel_decoder.py tests/test_query_head.py tests/test_prompt_builder_v2.py tests/test_hq_refiner.py tests/test_end2end_smoke.py tests/test_trainer_smoke.py tests/test_evaluator_metrics.py tests/test_infer_sliding_window.py -q`
- Decision: single trainer, warmup/joint curriculum, AMP path, resume, and named checkpoint flow passed

## S10

- Status: passed
- Time: 2026-04-04 04:34 UTC
- Scope check: evaluator, industrial metrics, threshold search, checkpoint threshold reuse helpers
- Test check:
  - `pytest tests/test_evaluator_metrics.py -q`
  - combined regression suite above
- Decision: evaluator and metrics path passed; threshold search writes reusable metadata

## S11

- Status: passed
- Time: 2026-04-04 04:34 UTC
- Scope check: sliding-window inference, cross-window fusion, final NMS, exporters
- Test check:
  - `pytest tests/test_infer_sliding_window.py -q`
  - combined regression suite above
- Decision: inferencer and exports passed in smoke form

## S12

- Status: passed
- Time: 2026-04-04 04:34 UTC
- Scope check: tests, smoke paths, resume, sliding-window inference coverage
- Test check:
  - `pytest tests/test_backbone_v2.py tests/test_pixel_decoder.py tests/test_query_head.py tests/test_prompt_builder_v2.py tests/test_hq_refiner.py tests/test_end2end_smoke.py tests/test_trainer_smoke.py tests/test_evaluator_metrics.py tests/test_infer_sliding_window.py -q`
- Decision: core regression suite passed

## S13

- Status: pending
- Time: 2026-04-04
- Scope check: overfit8 tooling is prepared via `scripts/run_overfit8.sh`
- Test check: not executed in the current environment
- Decision: blocked on actual long-running training acceptance
- Notes: current session has no CUDA device, so the real overfit8 acceptance run was not closed here

## S14

- Status: in progress
- Time: 2026-04-04
- Scope check: README, MigrationGuide, ReviewLog updated to new mainline
- Test check:
  - `python3 -m py_compile train_wirecr_hqinstsam.py eval_wirecr_hqinstsam.py infer_wirecr_hqinstsam.py`
- Decision: documentation updated, final close blocked by S13 acceptance
