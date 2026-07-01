# add_bids_json.ps1
# ==================
# Add BIDS JSON sidecar files to a converted BIDS dataset
#
# Creates:
#   - dataset_description.json
#   - task-rest_bold.json
#   - sub-*/ses-*/anat/sub-*_T1w.json (for each subject)
#
# Usage:
#   .\add_bids_json.ps1 -BIDSDir "D:\...\BIDS_batch"

param(
    [Parameter(Mandatory=$true)] [string]$BIDSDir
)

# 1. dataset_description.json
$dd = @{
    Name = "ABIDE CMU Batch"
    BIDSVersion = "1.8.0"
    DatasetType = "raw"
    Authors = @("ABIDE Consortium")
} | ConvertTo-Json

Set-Content -Path "$BIDSDir\dataset_description.json" -Value $dd
Write-Host "Created: dataset_description.json"

# 2. task-rest_bold.json
$tr = @{
    TaskName = "rest"
    RepetitionTime = 2.0
    Manufacturer = "Siemens"
    MagneticFieldStrength = 3.0
    Instructions = "Keep eyes open, fixate on crosshair"
} | ConvertTo-Json

Set-Content -Path "$BIDSDir\task-rest_bold.json" -Value $tr
Write-Host "Created: task-rest_bold.json"

# 3. T1w.json for each subject
$t1w = @{
    Modality = "MR"
    MagneticFieldStrength = 3.0
    Manufacturer = "Siemens"
    PulseSequenceType = "MPRAGE"
} | ConvertTo-Json

$subjects = Get-ChildItem -Path $BIDSDir -Directory -Filter "sub-*"
foreach ($subj in $subjects) {
    $jsonPath = Join-Path $subj.FullName "ses-01\anat\$($subj.Name)_ses-01_T1w.json"
    Set-Content -Path $jsonPath -Value $t1w
    Write-Host "Created: $($subj.Name) T1w.json"
}

Write-Host ""
Write-Host "All JSON sidecars added to $BIDSDir"