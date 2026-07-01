# run_fmriprep_single.ps1
# =========================
# Run fMRIPrep on a single subject
#
# Usage:
#   .\run_fmriprep_single.ps1 -BIDSDir "D:\...\BIDS" -OutDir "D:\...\output" -Subj "CMU50642" [-License "license.txt"]

param(
    [Parameter(Mandatory=$true)] [string]$BIDSDir,
    [Parameter(Mandatory=$true)] [string]$OutDir,
    [Parameter(Mandatory=$true)] [string]$Subj,
    [string]$License = "./license.txt"
)

$absLicense = (Resolve-Path $License).Path

docker run --rm `
    -v "${BIDSDir}:/data:ro" `
    -v "${OutDir}:/out" `
    -v "${absLicense}:/opt/freesurfer/license.txt" `
    nipreps/fmriprep:latest `
    /data /out participant `
    --participant-label $Subj `
    --output-spaces MNI152NLin2009cAsym:res-2 `
    --fs-no-reconall `
    --fs-license-file /opt/freesurfer/license.txt `
    -w /tmp/work `
    --nprocs 8 --mem 16000 `
    --skip-bids-validation