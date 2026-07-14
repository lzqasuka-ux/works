import os

d = r'c:\Users\zhongqing.lu\Desktop\works\works\Brain-Multimodal-Classification\2026.7.7\COBRE_BIDS\derivatives'
subs = sorted([x for x in os.listdir(d) if x.startswith('sub-') and os.path.isdir(os.path.join(d, x))])

done = []
partial = []

for s in subs:
    has_t1 = os.path.exists(os.path.join(d, s, 'ses-1', 'anat', f'{s}_ses-1_desc-preproc_T1w.nii.gz'))
    has_bold = os.path.exists(os.path.join(d, s, 'ses-1', 'func', f'{s}_ses-1_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'))
    if has_t1 and has_bold:
        done.append(s)
    elif has_t1 or has_bold:
        partial.append((s, has_t1, has_bold))

print(f'完全完成 (T1w + BOLD MNI): {len(done)}/{len(subs)}')
for x in done:
    print(f'  {x}')

print()
print(f'部分完成 (缺一项): {len(partial)}')
for s, t1, bl in partial:
    print(f'  {s}  T1w={"OK" if t1 else "MISS"}  BOLD={"OK" if bl else "MISS"}')

not_started = [s for s in subs if s not in done and s not in [p[0] for p in partial]]
if not_started:
    print(f'\n未开始: {len(not_started)} (from {not_started[0]} to {not_started[-1]})')