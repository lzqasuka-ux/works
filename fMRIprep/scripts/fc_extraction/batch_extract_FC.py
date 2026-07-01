"""
batch_extract_FC.py
===================
Batch extract AAL116 functional connectivity matrices from fMRIPrep outputs.

Input:  preproc_bold.nii.gz + confounds_timeseries.tsv (per subject)
Output: FC_matrix.csv (116×116, Fisher z) + FC_heatmap.png (per subject)

Dependencies: pip install nilearn pandas matplotlib seaborn

Usage:
    # Edit data_roots dict below with your paths, then:
    python batch_extract_FC.py
"""

import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from nilearn import datasets, image, maskers

# ============================================================
# CONFIGURATION — Edit paths for your dataset
# ============================================================
subjects = ["50642", "50643", "50644", "50645", "50646", "50647"]

data_roots = {
    "50642": r"D:\ABIDE_output\sub-CMU50642\ses-01\func",
    "50643": r"D:\ABIDE_output_batch\sub-CMU50643\ses-01\func",
    "50644": r"D:\ABIDE_output_batch\sub-CMU50644\ses-01\func",
    "50645": r"D:\ABIDE_output_batch\sub-CMU50645\ses-01\func",
    "50646": r"D:\ABIDE_output_batch\sub-CMU50646\ses-01\func",
    "50647": r"D:\ABIDE_output_batch\sub-CMU50647\ses-01\func",
}

out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output", "fc_results")
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# 1. Load AAL116 atlas
# ============================================================
print("Loading AAL atlas (SPM12, 116 ROIs) ...")
aal = datasets.fetch_atlas_aal(version="SPM12")
aal_img = aal.maps
aal_labels = list(aal.labels)[1:]  # drop background (label 0)
n_rois = len(aal_labels)
print(f"  Loaded {n_rois} ROIs")

# ============================================================
# 2. Process each subject
# ============================================================
for sub in subjects:
    root = data_roots[sub]
    bold_file = os.path.join(
        root,
        f"sub-CMU{sub}_ses-01_task-rest_"
        f"space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
    )
    conf_file = os.path.join(
        root,
        f"sub-CMU{sub}_ses-01_task-rest_desc-confounds_timeseries.tsv",
    )

    if not os.path.exists(bold_file) or not os.path.exists(conf_file):
        print(f"  sub-CMU{sub}: SKIPPED (missing file)")
        continue

    print(f"\nProcessing sub-CMU{sub} ...")

    # -- 2a. Load data
    bold_img = image.load_img(bold_file)
    conf_df = pd.read_csv(conf_file, sep="\t")

    # -- 2b. Select confound columns
    wanted_cols = []
    for prefix in [
        "trans_", "rot_", "a_comp_cor_", "t_comp_cor_",
        "csf", "white_matter", "global_signal",
    ]:
        for c in conf_df.columns:
            if c.startswith(prefix):
                wanted_cols.append(c)
    wanted_cols = list(dict.fromkeys(wanted_cols))
    confounds = conf_df[wanted_cols].copy()
    confounds = confounds.dropna(axis=1, how="any")
    print(f"  Using {confounds.shape[1]} confound columns")

    # -- 2c. Extract ROI time series + denoise
    masker = maskers.NiftiLabelsMasker(
        labels_img=aal_img,
        standardize="zscore_sample",
        strategy="mean",
        detrend=True,
        low_pass=0.08,
        high_pass=0.01,
        t_r=2.0,
    )
    roi_ts = masker.fit_transform(bold_img, confounds=confounds)
    print(f"  ROI time series shape: (T={roi_ts.shape[0]}, ROIs={roi_ts.shape[1]})")

    # -- 2d. Compute FC matrix
    fc = np.corrcoef(roi_ts.T)
    np.fill_diagonal(fc, 0)
    fc_z = np.arctanh(fc)  # Fisher z

    # -- 2e. Save CSV
    df_fc = pd.DataFrame(fc_z, columns=aal_labels, index=aal_labels)
    csv_path = os.path.join(out_dir, f"CMU_{sub}_FC_matrix.csv")
    df_fc.to_csv(csv_path)
    print(f"  -> {csv_path}")

    # -- 2f. Save heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        fc_z, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=False, yticklabels=False,
        square=True, cbar_kws={"label": "Fisher z"},
    )
    plt.title(f"CMU_{sub} — Rest FC (AAL116)", fontsize=14)
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"CMU_{sub}_FC_heatmap.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  -> {png_path}")

print("\n========== ALL DONE ==========")