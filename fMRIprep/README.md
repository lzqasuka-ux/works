# fMRIPrep Pipeline — ABIDE CMU Site

End-to-end fMRI preprocessing and functional connectivity (FC) extraction pipeline using fMRIPrep v25.2.5 + nilearn.

## Project Structure

```
fMRIPrep_pipeline/
├── README.md
├── requirements.txt                    # Python dependencies
├── .gitignore
├── config/
│   ├── dataset_description.json        # BIDS dataset metadata
│   ├── task-rest_bold.json             # fMRI acquisition parameters (TR, etc.)
│   └── templates/
│       └── sub-XXX_ses-01_T1w.json     # T1 sidecar template
├── scripts/
│   ├── bids_conversion/
│   │   └── convert_abide_to_bids.ps1   # ABIDE → BIDS batch converter
│   ├── fc_extraction/
│   │   └── batch_extract_FC.py         # AAL116 FC matrix + heatmap extraction
│   ├── run_fmriprep_single.ps1         # Run fMRIPrep on 1 subject
│   ├── run_fmriprep_batch.ps1          # Run fMRIPrep on multiple subjects
│   └── utils/
│       └── add_bids_json.ps1           # Add JSON sidecars to BIDS dir
├── docs/
│   └── pipeline_overview.md            # Full pipeline documentation
└── output/                             # (gitignored) Generated outputs
    └── fc_results/
```

## Pipeline Overview

```
ABIDE raw data                   fMRIPrep                    Post-processing
    │                               │                            │
    ▼                               ▼                            ▼
CMU_50643/                sub-CMU50643/                  FC_matrix.csv
  anat/NIfTI/           ┌── ses-01/anat/                 FC_heatmap.png
  │ mprage.nii.gz   ──► │     T1w.nii.gz
  rest/NIfTI/            └── ses-01/func/
    rest.nii.gz              task-rest_bold.nii.gz
                             confounds_timeseries.tsv
```

## Quick Start

### 1. Prerequisites

```bash
# Docker + fMRIPrep image
docker pull nipreps/fmriprep:latest

# FreeSurfer license (free registration)
# https://surfer.nmr.mgh.harvard.edu/registration.html

# Python packages
pip install -r requirements.txt
```

### 2. Convert ABIDE → BIDS

```powershell
.\scripts\bids_conversion\convert_abide_to_bids.ps1
.\scripts\utils\add_bids_json.ps1 -BIDSDir "D:\ABIDE\...\BIDS_batch"
```

### 3. Run fMRIPrep

```powershell
# Single subject (~16 min):
.\scripts\run_fmriprep_single.ps1 -BIDSDir "D:\...\BIDS" -OutDir "D:\...\output" -Subj "CMU50642" -License "license.txt"

# Batch (5 subjects, ~80 min):
.\scripts\run_fmriprep_batch.ps1 -BIDSDir "D:\...\BIDS_batch" -OutDir "D:\...\output_batch"
```

### 4. Extract FC Matrices

```bash
python scripts/fc_extraction/batch_extract_FC.py
```

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--output-spaces` | `MNI152NLin2009cAsym:res-2` | 2 mm MNI standard space |
| `--fs-no-reconall` | Enabled | Skip FreeSurfer surface reconstruction |
| `-w /tmp/work` | Linux internal path | **Critical**: avoid Windows path (`D:\`) in Docker |
| `--nprocs 8` | 8 CPU cores | Matches i7-14700HX HPC cores |
| `--mem 16000` | 16 GB | Safe limit for 32 GB system |
| `--skip-bids-validation` | Enabled | Skip TR mismatch warning |
| `t_r=2.0` | 2.0 s | ABIDE CMU TR |
| `low_pass=0.08, high_pass=0.01` | Hz | Resting-state bandpass filter |

## Dataset

- **Source**: ABIDE I, CMU site
- **Subjects**: CMU_50642–50647 (6 subjects)
- **Modalities**: T1w MPRAGE + resting-state fMRI
- **fMRIPrep version**: 25.2.5

## Outputs

### fMRIPrep (per subject)
- `*_desc-preproc_bold.nii.gz` — Fully preprocessed 4D BOLD (MNI space)
- `*_desc-confounds_timeseries.tsv` — Motion/CompCor/physiological regressors
- `*_desc-preproc_T1w.nii.gz` — Preprocessed T1 (native + MNI)
- `*_dseg.nii.gz` — Tissue segmentation (CSF/GM/WM)

### FC Extraction (per subject)
- `CMU_XXXXX_FC_matrix.csv` — 116×116 AAL FC matrix (Fisher z)
- `CMU_XXXXX_FC_heatmap.png` — Red-blue FC heatmap

## Known Issues

1. **AFNI Windows path error**: Always use `-w /tmp/work` (Linux path), never `D:\`
2. **TR mismatch**: ABIDE CMU NIfTI headers differ from JSON; use `--skip-bids-validation`
3. **AAL3 download blocked**: Company firewall blocks `gin.cnrs.fr`; use SPM12 version