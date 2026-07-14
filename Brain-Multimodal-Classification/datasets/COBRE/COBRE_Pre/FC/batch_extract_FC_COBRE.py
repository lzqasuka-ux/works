"""
batch_extract_FC_COBRE.py
=========================
Batch extract functional connectivity matrices from fMRIPrep outputs for COBRE dataset.
Uses three atlases: AAL116, Schaefer400, CC200.

Input:  preproc_bold.nii.gz + confounds_timeseries.tsv (per subject, MNI 2mm)
Output: FC_matrix.csv + FC_heatmap.png per atlas per subject

Dependencies: pip install nilearn pandas matplotlib seaborn

Usage:
    python batch_extract_FC_COBRE.py
"""

import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from nilearn import datasets, image, maskers

# ============================================================
# CONFIGURATION
# ============================================================
DERIV_DIR   = r"D:\数据集\COBRE\COBRE_BIDS\derivatives"
FC_BASE     = r"D:\数据集\COBRE\COBRE_FC"
SUBJECT_IDS = sorted(
    [x for x in os.listdir(DERIV_DIR) if x.startswith("sub-") and os.path.isdir(os.path.join(DERIV_DIR, x))]
)

# Atlases to extract
ATLASES = {
    "AAL116": {
        "dir":         os.path.join(FC_BASE, "AAL116"),
        "n_rois":      116,
        "load":        lambda: _load_aal_atlas(),
    },
    "Schaefer400": {
        "dir":         os.path.join(FC_BASE, "Schaefer400"),
        "n_rois":      400,
        "load":        lambda: _load_schaefer_atlas(),
    },
    "CC200": {
        "dir":         os.path.join(FC_BASE, "CC200"),
        "n_rois":      200,
        "load":        lambda: _load_cc200_atlas(),
    },
}

def _load_aal_atlas():
    fetched = datasets.fetch_atlas_aal(version="SPM12")
    img = image.load_img(fetched.maps)          # force load in case maps is str path
    labels = list(fetched.labels)[1:]           # drop background
    return img, labels

def _load_schaefer_atlas():
    fetched = datasets.fetch_atlas_schaefer_2018(n_rois=400)
    img = image.load_img(fetched.maps)          # force load in case maps is str path
    labels = list(fetched.labels)
    return img, labels

def _load_cc200_atlas():
    # CC200 官方下载 URL 有 SSL 证书问题, 直接使用本地已下载文件
    # 文件位置: C:\Users\zhongqing.lu\nilearn_data\craddock_2012\scorr05_mean_all.nii.gz
    cc200_nii = os.path.join(
        os.path.expanduser("~"), "nilearn_data", "craddock_2012",
        "scorr05_mean_all.nii.gz"
    )
    if not os.path.exists(cc200_nii):
        raise FileNotFoundError(f"CC200 atlas not found at {cc200_nii}. "
                                "Please manually download from NITRC.")
    # CC200 is a 4D prob map: (47,56,46,43) = 43 × probability map per ROI
    # Need to collapse to 3D label map via argmax
    import nibabel as nib
    nib_img = nib.load(cc200_nii)
    if nib_img.ndim == 4:
        prob_data = nib_img.get_fdata()                     # (47,56,46,43)
        label_data = np.argmax(prob_data, axis=3) + 1      # 0-based → 1-based labels
        label_data[np.all(prob_data == 0, axis=3)] = 0     # background = 0
        img = nib.Nifti1Image(label_data.astype(np.int16),
                              nib_img.affine, nib_img.header)
    else:
        img = image.load_img(cc200_nii)
    labels = [f"ROI_{i:03d}" for i in range(200)]
    return img, labels

# ============================================================
# 1. Prepare output directories
# ============================================================
for name, info in ATLASES.items():
    os.makedirs(info["dir"], exist_ok=True)

# ============================================================
# 2. Load and validate atlases
# ============================================================
print("Loading atlases ...")
atlas_data = {}
for name, info in ATLASES.items():
    print(f"  Loading {name} ...")
    img, labels = info["load"]()
    atlas_data[name] = {"img": img, "labels": labels}
    print(f"    -> shape={img.shape}, {len(labels)} ROIs loaded")

# ============================================================
# 3. Process each subject
# ============================================================

def _progress_bar(current, total, width=30):
    """简洁的文本进度条"""
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}| {pct*100:5.1f}%  ({current}/{total})"

print(f"\nProcessing {len(SUBJECT_IDS)} subjects ...")
total_subs = len(SUBJECT_IDS)
skipped = 0

for i, subj in enumerate(SUBJECT_IDS, 1):
    func_dir = os.path.join(DERIV_DIR, subj, "ses-1", "func")
    bold_file = os.path.join(
        func_dir,
        f"{subj}_ses-1_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
    )
    conf_file = os.path.join(
        func_dir,
        f"{subj}_ses-1_task-rest_desc-confounds_timeseries.tsv",
    )

    if not os.path.exists(bold_file):
        print(f"\n  {subj}: SKIP (missing BOLD)")
        skipped += 1
        continue
    if not os.path.exists(conf_file):
        print(f"\n  {subj}: SKIP (missing confounds)")
        skipped += 1
        continue

    bar = _progress_bar(i, total_subs)
    print(f"\n{bar}\n  {subj} ...")

    # --- 3a. Load data (once per subject)
    bold_img = image.load_img(bold_file)
    conf_df  = pd.read_csv(conf_file, sep="\t")

    # --- 3b. Select confound columns
    wanted_cols = []
    for prefix in [
        "trans_", "rot_", "a_comp_cor_", "t_comp_cor_",
        "csf", "white_matter", "global_signal",
    ]:
        for c in conf_df.columns:
            if c.startswith(prefix):
                wanted_cols.append(c)
    wanted_cols = list(dict.fromkeys(wanted_cols))          # keep order, dedup
    confounds = conf_df[wanted_cols].copy()
    confounds = confounds.dropna(axis=1, how="any")
    print(f"  Confound columns: {confounds.shape[1]}")

    # --- 3c. For each atlas, extract FC
    atlas_names = list(atlas_data.keys())
    for ai, atlas_name in enumerate(atlas_names, 1):
        ad = atlas_data[atlas_name]
        out_dir = ATLASES[atlas_name]["dir"]
        csv_path = os.path.join(out_dir, f"{subj}_FC_{atlas_name}.csv")
        png_path = os.path.join(out_dir, f"{subj}_FC_{atlas_name}.png")

        # Atlas-level mini progress
        print(f"  [{ai}/{len(atlas_names)}] {atlas_name} ...", end="", flush=True)

        # --- NiftiLabelsMasker
        masker = maskers.NiftiLabelsMasker(
            labels_img=ad["img"],
            strategy="mean",
            standardize="zscore_sample",
            detrend=True,
            low_pass=0.08,
            high_pass=0.01,
            t_r=2.0,
        )
        try:
            roi_ts = masker.fit_transform(bold_img, confounds=confounds)
        except Exception as e:
            print(f" FAILED ({e})")
            continue

        n_rois_extracted = roi_ts.shape[1]
        if n_rois_extracted != len(ad["labels"]):
            labels = [f"ROI_{i:03d}" for i in range(n_rois_extracted)]
        else:
            labels = ad["labels"]

        # --- FC matrix
        fc = np.corrcoef(roi_ts.T)
        np.fill_diagonal(fc, 0)
        fc_z = np.arctanh(fc)

        # --- Save CSV
        df_fc = pd.DataFrame(fc_z, columns=labels, index=labels)
        df_fc.to_csv(csv_path)

        # --- Save heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            fc_z, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            xticklabels=False, yticklabels=False,
            square=True, cbar_kws={"label": "Fisher z"},
        )
        plt.title(f"{subj} — Rest FC ({atlas_name})", fontsize=14)
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(" ✓")

print(f"\n{'='*60}")
print(f"ALL DONE — {total_subs - skipped} processed, {skipped} skipped, {total_subs} total")
print(f"{'='*60}")
