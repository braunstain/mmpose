$ErrorActionPreference = "Continue"

$config = "configs\body_2d_keypoint\rtmpose\coco\project_baseline.py"

$checkpoints = @(
   # "checkpoints\rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth",
   # "work_dirs\struct_lr_5e-4\best_coco_AP_epoch_7.pth",
   # "work_dirs\cropout_lr_1e-4\epoch_9.pth",
  #  "work_dirs\bone_lr_5e-4\best_coco_AP_epoch_10.pth",
   # "work_dirs\cropout_bone_lr_5e-4\epoch_14.pth"
   "work_dirs\only_ljc_lr_5e-4\best_coco_AP_epoch_10.pth",
   "work_dirs\ljc_struct_lr_5e-4\best_coco_AP_epoch_6.pth"

)
$annPairs = @(
    @{ ds = "annotations/val_split/ochuman_val_non_zero.json";  ev = "data/OCHuman/annotations/val_split/ochuman_val_non_zero.json"; id = "val_non_zero" },
    @{ ds = "annotations/val_split/ochuman_val_zero.json";      ev = "data/OCHuman/annotations/val_split/ochuman_val_zero.json"; id = "val_zero" },
    @{ ds = "annotations/val_split/ochuman_val_one.json";  ev = "data/OCHuman/annotations/val_split/ochuman_val_one.json"; id = "val_one" },
    @{ ds = "annotations/val_split/ochuman_val_more_than_one.json";  ev = "data/OCHuman/annotations/val_split/ochuman_val_more_than_one.json"; id = "val_more_than_one" },
    @{ ds = "annotations/ochuman_coco_format_val_range_0.00_1.00.json";  ev = "data/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json"; id = "val_general" },
    @{ ds = "annotations/test_split/ochuman_test_non_zero.json";  ev = "data/OCHuman/annotations/test_split/ochuman_test_non_zero.json"; id = "test_non_zero" },
    @{ ds = "annotations/test_split/ochuman_test_zero.json";      ev = "data/OCHuman/annotations/test_split/ochuman_test_zero.json"; id = "test_zero" },
    @{ ds = "annotations/test_split/ochuman_test_one.json";  ev = "data/OCHuman/annotations/test_split/ochuman_test_one.json"; id = "test_one" },
    @{ ds = "annotations/test_split/ochuman_test_more_than_one.json";  ev = "data/OCHuman/annotations/test_split/ochuman_test_more_than_one.json"; id = "test_more_than_one" },
    @{ ds = "annotations/ochuman_coco_format_test_range_0.00_1.00.json";  ev = "data/OCHuman/annotations/ochuman_coco_format_test_range_0.00_1.00.json"; id = "test_general" }
)


foreach ($ckpt in $checkpoints) {

        if (-not (Test-Path $ckpt)) {
            Write-Host "Skipping missing checkpoint: $ckpt" -ForegroundColor Yellow
            continue
        }

        foreach ($pair in $annPairs) {
                $dsAnn = $pair.ds
                $evAnn = $pair.ev
                $idAnn = $pair.id
            $cmd = @(
                "tools\test.py"
                $config
                $ckpt
                "--cfg-options"
                "test_dataloader.dataset.ann_file=$dsAnn"
                "test_evaluator.ann_file=$evAnn"
                "test_dataloader.num_workers=0"
                "test_dataloader.persistent_workers=False"
            )

            $output = & python @cmd 2>&1

            $textOutput = $output | Out-String

            $apMatch = [regex]::Match($textOutput, "AP\) @\[ IoU=0.50:0.95.*=\s+([0-9\.]+)")
            $arMatch = [regex]::Match($textOutput, "AR\) @\[ IoU=0.50:0.95.*=\s+([0-9\.]+)")

            $apValue = $apMatch.Groups[1].Value
            $arValue = $arMatch.Groups[1].Value
            Write-Host "Checkpoint : $ckpt, id: $idAnn" -ForegroundColor Magenta
            if ($apValue) {
                Write-Host "AP: $apValue" -ForegroundColor Green
            }
                        
            else {
                Write-Host "AP line not found" -ForegroundColor Red
            }

            if ($arValue) {
                Write-Host "AR: $arValue" -ForegroundColor Green
            } 
            else {
                Write-Host "AR line not found" -ForegroundColor Red
            }


            if (-not (Test-Path "eval_summary_final.csv")) {
                "method,subset,AP,AR" | Out-File "eval_summary_final.csv"
            }

            Add-Content "eval_summary_final.csv" "$ckpt,$idAnn,$apValue,$arValue"
        }

}
    

Write-Host ""
Write-Host "Done. Summary saved to $summaryLog" -ForegroundColor Magenta