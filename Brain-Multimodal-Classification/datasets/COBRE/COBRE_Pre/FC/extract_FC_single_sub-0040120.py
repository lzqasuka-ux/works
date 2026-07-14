"""单独对 sub-0040120 重新提取 Schaefer400 FC"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from nilearn import datasets, image, maskers

DERIV_DIR = r"D:\数据集\COBRE\COBRE_BIDS\derivatives"
FC_DIR    = r"D:\数据集\COBRE\COBRE_FC\Schaefer400"
SUBJ      = "sub-0040120"

# 加载 Schaefer400 图谱
print("Loading Schaefer400 atlas ...")
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400)
atlas_img = image.load_img(atlas.maps)
atlas_labels = list(atlas.labels)
print(f"  shape={atlas_img.shape}, {len(atlas_labels)} ROIs")

# 受试者文件
func_dir = os.path.join(DERIV_DIR, SUBJ, "ses-1", "func")
bold_file = os.path.join(func_dir, f"{SUBJ}_ses-1_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
conf_file = os.path.join(func_dir, f"{SUBJ}_ses-1_task-rest_desc-confounds_timeseries.tsv")

if not os.path.exists(bold_file):
    print(f"ERROR: missing BOLD at {bold_file}")
    exit(1)
if not os.path.exists(conf_file):
    print(f"ERROR: missing confounds at {conf_file}")
    exit(1)

print("Loading BOLD + confounds ...")
bold_img = image.load_img(bold_file)
conf_df  = pd.read_csv(conf_file, sep="\t")

# 选择 confound 列
wanted_cols = []
for prefix in ["trans_", "rot_", "a_comp_cor_", "t_comp_cor_", "csf", "white_matter", "global_signal"]:
    for c in conf_df.columns:
        if c.startswith(prefix):
            wanted_cols.append(c)
wanted_cols = list(dict.fromkeys(wanted_cols))
confounds = conf_df[wanted_cols].dropna(axis=1, how="any")
print(f"  Confound columns: {confounds.shape[1]}")

# NiftiLabelsMasker
masker = maskers.NiftiLabelsMasker(
    labels_img=atlas_img,
    strategy="mean",
    standardize="zscore_sample",
    detrend=True,
    low_pass=0.08,
    high_pass=0.01,
    t_r=2.0,
)
print("Extracting ROI time series ...")
roi_ts = masker.fit_transform(bold_img, confounds=confounds)
n_rois = roi_ts.shape[1]
print(f"  shape: {roi_ts.shape[0]} timepoints × {n_rois} ROIs")

# FC matrix
fc = np.corrcoef(roi_ts.T)
np.fill_diagonal(fc, 0)
fc_z = np.arctanh(fc)

# 输出
out_dir = os.path.join(FC_DIR, f"{SUBJ}_FC")
os.makedirs(out_dir, exist_ok=True)

csv_path = os.path.join(out_dir, f"{SUBJ}_FC_Schaefer400.csv")
png_path = os.path.join(out_dir, f"{SUBJ}_FC_Schaefer400.png")

df_fc = pd.DataFrame(fc_z, columns=atlas_labels[:n_rois], index=atlas_labels[:n_rois])
df_fc.to_csv(csv_path)
print(f"  -> {csv_path}")

plt.figure(figsize=(14, 12))
sns.heatmap(fc_z, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            xticklabels=False, yticklabels=False,
            square=True, cbar_kws={"label": "Fisher z"})
plt.title(f"{SUBJ} — Rest FC (Schaefer400)", fontsize=14)
plt.tight_layout()
plt.savefig(png_path, dpi=150)
plt.close()
print(f"  -> {png_path}")

print("Done. FC extracted successfully.")