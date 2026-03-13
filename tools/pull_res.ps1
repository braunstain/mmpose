$ErrorActionPreference = "Continue"

$config = "configs\body_2d_keypoint\rtmpose\coco\project_baseline.py"

$expDirs = @(
    "work_dirs\struct_lr_5e-4"
)
$annPairs = @(
    @{ ds = "annotations/val_split/ochuman_non_zero.json";  ev = "data/OCHuman/annotations/val_split/ochuman_non_zero.json" },
    @{ ds = "annotations/val_split/ochuman_zero.json";      ev = "data/OCHuman/annotations/val_split/ochuman_zero.json" },
    @{ ds = "annotations/test_split/ochuman_non_zero.json"; ev = "data/OCHuman/annotations/test_split/ochuman_non_zero.json" },
    @{ ds = "annotations/test_split/ochuman_zero.json";     ev = "data/OCHuman/annotations/test_split/ochuman_zero.json" }
)


foreach ($expDir in $expDirs) {
    foreach ($epoch in 1..20) {
        $ckpt = Join-Path $expDir "epoch_$epoch.pth"

        if (-not (Test-Path $ckpt)) {
            Write-Host "Skipping missing checkpoint: $ckpt" -ForegroundColor Yellow
            continue
        }

        foreach ($pair in $annPairs) {
                $dsAnn = $pair.ds
                $evAnn = $pair.ev
                Write-Host "EPOCH : $epoch" -ForegroundColor Cyan
                Write-Host "DATASET ANN   : $dsAnn" -ForegroundColor Cyan
                Write-Host "EVALUATOR ANN : $evAnn" -ForegroundColor Cyan

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

            $lr = ($expDir -replace "work_dirs\\struct_lr_5e-4","")

            $split = if ($dsAnn -match "val_split") {"val"} else {"test"}
            $occ   = if ($dsAnn -match "non_zero") {"occ"} else {"no_occ"}

            if (-not (Test-Path "eval_summary_struct_lr_5e-4.csv")) {
                "lr,epoch,split,occ,AP,AR" | Out-File "eval_summary_struct_lr_5e-4.csv"
            }

            Add-Content "eval_summary_struct_lr_5e-4.csv" "$lr,$epoch,$split,$occ,$apValue,$arValue"
            }
        }

    
}
    

Write-Host ""
Write-Host "Done. Summary saved to $summaryLog" -ForegroundColor Magenta