for %%L in (1e-2 5e-2) do (
  python tools/train.py configs/body_2d_keypoint/rtmpose/coco/no_change_cfg.py ^
  --cfg-options optim_wrapper.optimizer.lr=%%L ^
  --work-dir work_dirs/no_change_lr_%%L
)