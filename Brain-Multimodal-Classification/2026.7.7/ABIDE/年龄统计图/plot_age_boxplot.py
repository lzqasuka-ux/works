import csv, os, matplotlib.pyplot as plt, matplotlib.patches as mpatches
base = os.path.dirname(__file__)

def safe_read(fp):
    for enc in ['utf-8','gbk','latin-1','cp1252']:
        try:
            with open(fp,'r',encoding=enc) as f: return list(csv.DictReader(f)),enc
        except: continue
    with open(fp,'r',encoding='utf-8',errors='replace') as f: return list(csv.DictReader(f)),'replace'

r1,_ = safe_read(os.path.join(base,'ABIDE-1','Phenotypic_V1_0b.csv'))
r2,_ = safe_read(os.path.join(base,'ABIDE-2','ABIDEII_Composite_Phenotypic.csv'))

ages = {}
for row in r1:
    dx = row.get('DX_GROUP',''); age = row.get('AGE_AT_SCAN','')
    try: age = float(age)
    except: continue
    label = 'ASD' if dx=='1' else 'HC' if dx=='2' else None
    if label: ages.setdefault(label,[]).append(age)
for row in r2:
    dx = row.get('DX_GROUP',''); age = row.get('AGE_AT_SCAN ','')
    try: age = float(age)
    except: continue
    label = 'ASD' if dx=='1' else 'HC' if dx=='2' else None
    if label: ages.setdefault(label,[]).append(age)

order = ['ASD','HC']
colors_box = ['#E76F51','#2A9D8F']
data = [ages[k] for k in order]

fig,ax = plt.subplots(figsize=(5.2,5))
bp = ax.boxplot(data, patch_artist=True, widths=0.4, showfliers=True,
                flierprops={'marker':'o','markersize':4,'markerfacecolor':'gray','alpha':0.5})
for patch, c in zip(bp['boxes'], colors_box): patch.set_facecolor(c)
for median in bp['medians']: median.set_color('white')

# ===== 隐藏默认 x 轴刻度标签，用 ax.text 在 axes 相对坐标放置 =====
ax.set_xticks([1, 2])
ax.set_xticklabels([])
ax.set_ylabel('Age (years)', fontsize=12, fontweight='bold', labelpad=8)
ax.set_title('ABIDE - Age Distribution by Diagnosis', fontsize=13, fontweight='bold', pad=16)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=11)

ymax = max(max(d) for d in data)
ax.set_ylim(0, ymax * 1.15)

# ----- 核心：用 transform=ax.transAxes 在 axes 坐标系放置标签 -----
# (0, 0) = 左下角, (1, 0) = 右下角, 下面就是紧贴 x 轴下方
# y_offset 改为负数（单位：axes 高度的倍数），越负越靠下
y_offset = -0.05   # ← 改这个！ -0.03=靠近柱子, -0.08=更远
trans = ax.get_xaxis_transform()  # x: 数据坐标, y: axes 相对坐标
for i, (label, color) in enumerate(zip(order, colors_box)):
    ax.text(i + 1, y_offset, label, ha='center', va='top',
            fontsize=12, fontweight='bold', color=color,
            transform=trans)

leg = [mpatches.Patch(color='#E76F51',label='ASD (n=%d)'%len(ages['ASD'])),
       mpatches.Patch(color='#2A9D8F',label='HC (n=%d)'%len(ages['HC']))]
ax.legend(handles=leg, loc='upper right', frameon=False, fontsize=9.5, handlelength=1.2)

fig.tight_layout(pad=1.5)
out = os.path.join(base, 'ABIDE_age_boxplot.png')
plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.15); plt.close()
print('ABIDE_age_boxplot.png saved.')