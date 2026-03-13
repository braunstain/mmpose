for %%L in (5e-4) do (
  python tools/train.py configs/body_2d_keypoint/rtmpose/coco/struct_cfg.py ^
  --cfg-options optim_wrapper.optimizer.lr=%%L ^
  --work-dir work_dirs/struct_lr_%%L_lw_0.5
)