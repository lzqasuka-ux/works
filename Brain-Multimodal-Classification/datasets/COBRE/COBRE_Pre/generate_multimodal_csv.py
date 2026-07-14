"""
生成 COBRE 多模态训练 CSV，列：
  ID, Sex, Age, Label, sMRI_path, FC_path

从 COBRE_phenotypic_data.csv 做 ID 映射，只包含有配对文件的受试者
"""

import os, sys, pandas as pd

# 1. Read phenotypic data
pheno_path = r"C:\Users\zhongqing.lu\Desktop\works\数据集\COBRE\COBRE_phenotypic_data.csv"
pheno = pd.read_csv(pheno_path)
print(f"Phenotypic data: {len(pheno)} subjects")

# Rename first column to ID
pheno.rename(columns={pheno.columns[0]: 'ID'}, inplace=True)

# Filter out disenrolled subjects
disenrolled_mask = (
    (pheno['Subject Type'].str.lower() == 'disenrolled') |
    (pheno['Current Age'].astype(str).str.lower() == 'disenrolled')
)
print(f"Removing {disenrolled_mask.sum()} disenrolled subjects")
pheno = pheno[~disenrolled_mask].copy()

# Map ID (40000-40147 → sub-0040000 → sub-0040147)
pheno['SubjID'] = pheno['ID'].apply(lambda x: f"sub-{int(x):07d}")

# Label: Control → 0, Patient → 1
pheno['Label'] = pheno['Subject Type'].apply(lambda x: 0 if str(x).strip().lower() == 'control' else 1)
print(f"Labels: {pheno['Label'].value_counts().to_dict()}")

# Sex: Male → 1, Female → 0
sex_map = {'male': 1, 'female': 0}
pheno['Sex'] = pheno['Gender'].str.strip().str.lower().map(sex_map)
sex_nulls = pheno['Sex'].isna().sum()
if sex_nulls > 0:
    print(f"Warning: {sex_nulls} subjects have unknown sex, setting to -1")
    pheno['Sex'] = pheno['Sex'].fillna(-1)

# Age
pheno['Age'] = pd.to_numeric(pheno['Current Age'], errors='coerce')
age_nulls = pheno['Age'].isna().sum()
if age_nulls > 0:
    print(f"Warning: {age_nulls} subjects have unknown age, dropping them")
    pheno = pheno.dropna(subset=['Age'])
    pheno['Age'] = pheno['Age'].astype(int)

# Build sMRI and FC paths from COBRE_LOAD
load_base = r"D:\数据集\COBRE\COBRE_LOAD"

def get_smri_path(subj_id):
    p = os.path.join(load_base, subj_id, f"{subj_id}_ses-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz")
    return p if os.path.exists(p) else None

def get_fc_path(subj_id):
    p = os.path.join(load_base, subj_id, f"{subj_id}_FC_Schaefer400.csv")
    return p if os.path.exists(p) else None

# Validate file existence
pheno['sMRI_path'] = pheno['SubjID'].apply(get_smri_path)
pheno['FC_path'] = pheno['SubjID'].apply(get_fc_path)

missing_smri = pheno['sMRI_path'].isna().sum()
missing_fc = pheno['FC_path'].isna().sum()
print(f"Missing sMRI: {missing_smri}, Missing FC: {missing_fc}")

# Keep only subjects with both files
pheno = pheno.dropna(subset=['sMRI_path', 'FC_path']).copy()
print(f"After filtering missing files: {len(pheno)} subjects")

# Final output columns
out_df = pheno[['SubjID', 'Sex', 'Age', 'Label', 'sMRI_path', 'FC_path']].copy()
out_df.columns = ['ID', 'Sex', 'Age', 'Label', 'sMRI_path', 'FC_path']

# Save
out_path = os.path.join(os.path.dirname(__file__), "..", "COBRE_multimodal.csv")
out_path = os.path.abspath(out_path)
out_df.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}  ({len(out_df)} subjects)")
print()
print("Column description:")
print("  ID         - Subject identifier (sub-XXXXXXX)")
print("  Sex        - 0=Female, 1=Male")
print("  Age        - Current age at scan")
print("  Label      - 0=Control, 1=Patient (Schizophrenia)")
print("  sMRI_path  - Preprocessed T1w in MNI space (2mm)")
print("  FC_path    - Schaefer400 FC matrix (Fisher z, 400x400)")

# Overview statistics
print(f"\nDataset summary:")
print(f"  Controls: {(out_df['Label']==0).sum()}")
print(f"  Patients: {(out_df['Label']==1).sum()}")
print(f"  Age range: {out_df['Age'].min()} - {out_df['Age'].max()}")
print(f"  Female: {(out_df['Sex']==0).sum()}, Male: {(out_df['Sex']==1).sum()}")