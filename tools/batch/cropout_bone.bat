for %%L in (5e-4) do (
  python tools/train.py configs/body_2d_keypoint/rtmpose/coco/cropout_bone_cfg.py ^
  --cfg-options optim_wrapper.optimizer.lr=%%L ^
  --work-dir work_dirs/cropout_bone_lr_%%L
)