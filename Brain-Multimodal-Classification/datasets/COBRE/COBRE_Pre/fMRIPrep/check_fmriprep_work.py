import os

d = r'D:\数据集\COBRE\fmriprep_work\fmriprep_25_2_wf'

if not os.path.exists(d):
    print('目录不存在')
    exit()

top = os.listdir(d)
print(f'顶层条目数: {len(top)}')
print(f'样本: {sorted(top)[:10]}')
print()

# 计算总大小
total_size = 0
for dp, _, files in os.walk(d):
    for f in files:
        fp = os.path.join(dp, f)
        if os.path.isfile(fp):
            total_size += os.path.getsize(fp)
print(f'总大小: {total_size / (1024**3):.2f} GB')

# 受试者工作目录
subjs = [x for x in top if x.startswith('sub_')]
print(f'\n受试者工作目录: {len(subjs)} 个')
if subjs:
    print(f'样本受试者目录: {sorted(subjs)[:3]}')
    # 检查第一个
    sub0 = os.path.join(d, subjs[0])
    print(f'\n{subjs[0]} 包含的子目录:')
    for x in sorted(os.listdir(sub0)):
        xp = os.path.join(sub0, x)
        if os.path.isdir(xp):
            file_cnt = sum(1 for _, _, files in os.walk(xp) for _ in files)
            print(f'  {x}/ ({file_cnt} 文件)')
        else:
            print(f'  {x} (文件)')