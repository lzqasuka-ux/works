"""
重新生成 COBRE_multimodal 数据，同时输出 CSV / JSON / Parquet 三种格式。
路径使用相对于 COBRE 目录的相对路径，解决跨机器使用问题。
"""

import pandas as pd
import json
import os

BASE = r"d:\Datasets\COBRE"

# 1. 读取 phenotypic data
pheno_path = r"C:\Users\zhongqing.lu\Desktop\works\数据集\COBRE\COBRE_phenotypic_data.csv"
pheno = pd.read_csv(pheno_path)
pheno.rename(columns={pheno.columns[0]: "ID"}, inplace=True)

# 2. 过滤 disenrolled
mask = (
    (pheno["Subject Type"].str.lower() == "disenrolled")
    | (pheno["Current Age"].astype(str).str.lower() == "disenrolled")
)
pheno = pheno[~mask].copy()

# 3. 构建 sub- ID
pheno["SubjID"] = pheno["ID"].apply(lambda x: f"sub-{int(x):07d}")

# 4. Label / Sex / Age
pheno["Label"] = pheno["Subject Type"].apply(
    lambda x: 0 if str(x).strip().lower() == "control" else 1
)
sex_map = {"male": 1, "female": 0}
pheno["Sex"] = (
    pheno["Gender"].str.strip().str.lower().map(sex_map).fillna(-1).astype(int)
)
pheno["Age"] = pd.to_numeric(pheno["Current Age"], errors="coerce").fillna(0).astype(int)

# 5. 构建相对路径（相对于 BASE）
pheno["sMRI_path"] = pheno["SubjID"].apply(
    lambda sid: os.path.join(
        "COBRE_LOAD", sid,
        f"{sid}_ses-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz",
    )
)
pheno["FC_path"] = pheno["SubjID"].apply(
    lambda sid: os.path.join(
        "COBRE_LOAD", sid, f"{sid}_FC_Schaefer400.csv"
    )
)

# 6. 只保留文件实际存在的受试者
def exists(rel_path):
    return os.path.exists(os.path.join(BASE, rel_path))

pheno = pheno[
    pheno["sMRI_path"].apply(exists) & pheno["FC_path"].apply(exists)
].copy()

print(f"File-verified subjects: {len(pheno)}")

# 7. 输出列
out = pheno[["SubjID", "Sex", "Age", "Label", "sMRI_path", "FC_path"]].copy()
out.columns = ["ID", "Sex", "Age", "Label", "sMRI_path", "FC_path"]

# ============================================================
# 输出三种格式
# ============================================================

# --- CSV (UTF-8 BOM, 兼容 Excel) ---
csv_path = os.path.join(BASE, "COBRE_multimodal.csv")
out.to_csv(csv_path, index=False, encoding="utf-8-sig")

# --- JSON (可读性好，记事本/VS Code 流畅打开) ---
records = out.to_dict(orient="records")
json_path = os.path.join(BASE, "COBRE_multimodal.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

# --- Parquet (列式存储，加载最快，体积最小，PyTorch 首选) ---
pq_path = os.path.join(BASE, "COBRE_multimodal.parquet")
out.to_parquet(pq_path, index=False)

# --- 验证 ---
df_csv = pd.read_csv(csv_path)
df_pq = pd.read_parquet(pq_path)
assert len(df_csv) == len(df_pq) == len(out), "Row count mismatch!"

csv_size = os.path.getsize(csv_path)
json_size = os.path.getsize(json_path)
pq_size = os.path.getsize(pq_path)

print()
print("=" * 55)
print("  格式对比")
print("=" * 55)
print(f"  CSV:     {csv_size/1024:>6.1f} KB   (Excel 打开，长路径会卡)")
print(f"  JSON:    {json_size/1024:>6.1f} KB   (记事本/VS Code 流畅)")
print(f"  Parquet: {pq_size/1024:>6.1f} KB   (训练加载最快)")
print("=" * 55)
print()
print("  Python 加载:")
print("    JSON:    df = pd.read_json('COBRE_multimodal.json')")
print("    Parquet: df = pd.read_parquet('COBRE_multimodal.parquet')")
print("    CSV:     df = pd.read_csv('COBRE_multimodal.csv')")
print()
print("  推荐: 训练用 Parquet，查看用 JSON，别再 Excel 开 CSV 了 😅")
print()
print(f"  路径基准: {BASE}")
print(f"  示例 sMRI_path: {out.iloc[0]['sMRI_path']}")
print(f"  示例 FC_path:   {out.iloc[0]['FC_path']}")