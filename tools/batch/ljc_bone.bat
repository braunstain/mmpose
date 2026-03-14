for %%L in (5e-4) do (
  python tools/train.py configs/body_2d_keypoint/rtmpose/coco/ljc_bone_cfg.py ^
  --cfg-options optim_wrapper.optimizer.lr=%%L ^
  --work-dir work_dirs/ljc_bone_lr_%%L
)