"""将 COBRE 原始数据转换为 BIDS 格式"""
import os, shutil, json

SRC = r'C:\Users\zhongqing.lu\Desktop\works\数据集\COBRE\COBRE'
DST = r'c:\Users\zhongqing.lu\Desktop\works\works\Brain-Multimodal-Classification\2026.7.7\COBRE_BIDS'

os.makedirs(DST, exist_ok=True)

# 写 dataset_description.json
desc = {
    "Name": "COBRE",
    "BIDSVersion": "1.8.0",
    "DatasetType": "raw",
    "Authors": ["COBRE Consortium"],
    "Description": "COBRE rs-fMRI dataset converted from raw NIfTI scans."
}
with open(os.path.join(DST, 'dataset_description.json'), 'w') as f:
    json.dump(desc, f, indent=2)

subjects = sorted([x for x in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, x))])
print(f'Found {len(subjects)} subjects. Converting...')

for i, subj in enumerate(subjects):
    # BIDS 目录
    bids_sub = f'sub-{subj}'
    anat_dir = os.path.join(DST, bids_sub, 'ses-1', 'anat')
    func_dir = os.path.join(DST, bids_sub, 'ses-1', 'func')
    os.makedirs(anat_dir, exist_ok=True)
    os.makedirs(func_dir, exist_ok=True)

    # 源文件路径
    src_anat = os.path.join(SRC, subj, 'session_1', 'anat_1', 'mprage.nii.gz')
    src_func = os.path.join(SRC, subj, 'session_1', 'rest_1', 'rest.nii.gz')

    # BIDS 文件名
    dst_anat = os.path.join(anat_dir, f'{bids_sub}_ses-1_T1w.nii.gz')
    dst_func = os.path.join(func_dir, f'{bids_sub}_ses-1_task-rest_bold.nii.gz')

    if os.path.exists(src_anat):
        shutil.copy2(src_anat, dst_anat)
    else:
        print(f'  WARNING: missing anat for {subj}')

    if os.path.exists(src_func):
        shutil.copy2(src_func, dst_func)
    else:
        print(f'  WARNING: missing func for {subj}')

    if (i + 1) % 20 == 0:
        print(f'  {i+1}/{len(subjects)} done...')

print(f'Done. BIDS dataset at: {DST}')