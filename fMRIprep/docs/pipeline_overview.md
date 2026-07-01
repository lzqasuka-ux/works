# fMRIPrep Preprocessing Pipeline — Full Documentation

## 1. Data Source

- Dataset: ABIDE I (Autism Brain Imaging Data Exchange)
- Site: CMU (Carnegie Mellon University)
- N = 6 (CMU_50642 through CMU_50647)
- Modalities: T1w MPRAGE + resting-state fMRI (eyes open, fixate crosshair)
- Scanner: Siemens 3T
- TR = 2.0 s

## 2. Preprocessing Steps (fMRIPrep v25.2.5)

### 2.1 Anatomical (T1w)

| Step | Tool | Description |
|------|------|-------------|
| N4 Bias Correction | ANTs N4BiasFieldCorrection | Correct B1 field inhomogeneity |
| Skull Stripping | ANTs antsBrainExtraction | Remove non-brain tissue |
| Tissue Segmentation | FSL FAST | GM, WM, CSF probability maps |
| MNI Normalization | ANTs SyN | Nonlinear registration to MNI152NLin2009cAsym |

### 2.2 Functional (BOLD)

| Step | Tool | Description |
|------|------|-------------|
| Reference Generation | RobustAverage | Motion-free mean BOLD reference |
| Head Motion Correction | FSL MCFLIRT | 6-DOF rigid-body realignment |
| Brain Extraction | Niworkflows (N4 + BET + dilation) | BOLD-space brain mask |
| EPI → T1 Coregistration | FreeSurfer BBR + FSL FLIRT | Boundary-based registration |
| MNI Resampling | ANTs + nitransforms | One-shot cubic B-spline interpolation |
| Confound Extraction | aCompCor + tCompCor | WM/CSF PCA + global signals + FD/DVARS |

### 2.3 Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| FreeSurfer | OFF (`--fs-no-reconall`) | No surface reconstruction |
| Output Space | MNI152NLin2009cAsym, 2 mm | ICBM 152 nonlinear asymmetric 2009c |
| Field Map | Not available | SyN-SDC or simple BBR coregistration |
| Slice Timing | Skipped | TR mismatch with BIDS JSON |

## 3. Functional Connectivity Extraction

### 3.1 Pipeline

1. Load AAL116 atlas (SPM12, 116 ROIs)
2. For each subject:
   - Load `preproc_bold.nii.gz` (MNI 2mm, preprocessed)
   - Load `confounds_timeseries.tsv`
   - Select key regressors: 6 DOF motion + aCompCor + tCompCor + CSF/WM/global signals
   - Drop NaN columns
   - NiftiLabelsMasker: ROI-mean extraction + confound regression + detrend + bandpass (0.01–0.08 Hz) + z-score
   - `np.corrcoef` → 116×116 Pearson r matrix
   - `np.arctanh` → Fisher z transformation
   - Save CSV (with AAL region labels) + heatmap PNG

### 3.2 Outputs

- `CMU_XXXXX_FC_matrix.csv`: 116 × 116 Fisher z matrix
- `CMU_XXXXX_FC_heatmap.png`: Red-blue heatmap (RdBu_r colormap)

## 4. Runtime Estimates

| Stage | Hardware | Time |
|-------|----------|------|
| fMRIPrep (1 subject) | i7-14700HX, 8 cores, 32 GB | ~16 min |
| fMRIPrep (5 subjects, batch) | same | ~80 min |
| FC extraction (6 subjects) | any | ~5 min |

## 5. Known Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `Can't open dataset D:/...` in AFNI | Windows path passed to Docker | Use `-w /tmp/work` only |
| `KeyError: 'RepetitionTime'` | Missing `task-rest_bold.json` TR field | Add `"RepetitionTime": 2.0` to JSON |
| `REPETITION_TIME_MISMATCH` | NIfTI header ≠ JSON TR | Use `--skip-bids-validation` |
| AAL3 download timeout | Company firewall blocks `gin.cnrs.fr` | Use `version="SPM12"` (OSF mirror) |
| `ValueError: array must not contain infs or NaNs` | Confound columns with NaN | Add `dropna(axis=1)` before `fit_transform()` |
| `Shape of passed values is (116,116), indices imply (117,117)` | AAL includes background label 0 | Use `list(aal.labels)[1:]` |