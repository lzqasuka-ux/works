# convert_abide_to_bids.ps1
# ============================
# Batch convert ABIDE CMU site data to BIDS format
#
# Input format (ABIDE):
#   CMU_50643/
#     anat/NIfTI/mprage.nii.gz
#     rest/NIfTI/rest.nii.gz
#
# Output format (BIDS):
#   BIDS_batch/
#     sub-CMU50643/ses-01/anat/sub-CMU50643_ses-01_T1w.nii.gz
#     sub-CMU50643/ses-01/func/sub-CMU50643_ses-01_task-rest_bold.nii.gz
#
# Usage:
#   1. Edit $subjects and $srcBase below
#   2. Run: .\convert_abide_to_bids.ps1
#   3. Then add JSON metadata (see ../config/ templates)

$subjects = @("50643", "50644", "50645", "50646", "50647")  # Change this list
$srcBase = "D:\ABIDE\lzqzs-20260623_201850"                  # Source ABIDE data
$bidsBase = "D:\ABIDE\lzqzs-20260623_201850\BIDS_batch"      # Output BIDS dir

New-Item -ItemType Directory -Force -Path $bidsBase | Out-Null

foreach ($s in $subjects) {
    $ses = "ses-01"
    $anatDir = "$bidsBase\sub-CMU$s\$ses\anat"
    $funcDir = "$bidsBase\sub-CMU$s\$ses\func"

    New-Item -ItemType Directory -Force -Path $anatDir | Out-Null
    New-Item -ItemType Directory -Force -Path $funcDir | Out-Null

    Copy-Item "$srcBase\CMU_$s\anat\NIfTI\mprage.nii.gz" `
        -Destination "$anatDir\sub-CMU$($s)_${ses}_T1w.nii.gz"

    Copy-Item "$srcBase\CMU_$s\rest\NIfTI\rest.nii.gz" `
        -Destination "$funcDir\sub-CMU$($s)_${ses}_task-rest_bold.nii.gz"

    Write-Host "Converted: sub-CMU$s"
}

Write-Host ""
Write-Host "All subjects converted to BIDS in: $bidsBase"
Write-Host "Next: add JSON metadata files (dataset_description.json, task-rest_bold.json, T1w.json)"