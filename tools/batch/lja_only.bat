for %%L in (1e-2 1e-3 1e-4) do (
  python tools/train.py configs/body_2d_keypoint/rtmpose/coco/ljc_cfg.py ^
  --cfg-options optim_wrapper.optimizer.lr=%%L ^
  --work-dir work_dirs/only_ljc_05_lr_%%L
)