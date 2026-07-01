# run_fmriprep_batch.ps1
# ========================
# Batch run fMRIPrep on multiple subjects
#
# Usage:
#   .\run_fmriprep_batch.ps1 -BIDSDir "D:\...\BIDS_batch" -OutDir "D:\...\output_batch" [-License "license.txt"]

param(
    [Parameter(Mandatory=$true)] [string]$BIDSDir,
    [Parameter(Mandatory=$true)] [string]$OutDir,
    [string]$License = "./license.txt",
    [string[]]$Subjects = @("CMU50643", "CMU50644", "CMU50645", "CMU50646", "CMU50647")
)

$absLicense = (Resolve-Path $License).Path
$subjList = $Subjects -join " "

Write-Host "Running fMRIPrep on subjects: $subjList"
Write-Host ""

docker run --rm `
    -v "${BIDSDir}:/data:ro" `
    -v "${OutDir}:/out" `
    -v "${absLicense}:/opt/freesurfer/license.txt" `
    nipreps/fmriprep:latest `
    /data /out participant `
    --participant-label $($Subjects -join " ") `
    --output-spaces MNI152NLin2009cAsym:res-2 `
    --fs-no-reconall `
    --fs-license-file /opt/freesurfer/license.txt `
    -w /tmp/work `
    --nprocs 8 --mem 16000 `
    --skip-bids-validation

Write-Host ""
Write-Host "Done. Outputs saved to: $OutDir"