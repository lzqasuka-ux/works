import csv, os, matplotlib.pyplot as plt, matplotlib.patches as mpatches
base = os.path.dirname(__file__)

def safe_read(fp):
    for enc in ['utf-8','gbk','latin-1','cp1252']:
        try:
            with open(fp,'r',encoding=enc) as f: return list(csv.DictReader(f)),enc
        except: continue
    with open(fp,'r',encoding='utf-8',errors='replace') as f: return list(csv.DictReader(f)),'replace'

# CSV 在上一层 PPMI 目录
csv_path = os.path.join(os.path.dirname(base), 'PPMIALL_7_07_2026.csv')
rows,_ = safe_read(csv_path)
seen = {}
for row in rows:
    sid = row.get('Subject',''); grp = row.get('Group',''); age = row.get('Age','')
    if sid and sid not in seen:
        try:
            if grp == 'Control': grp = 'HC'  # CSV 中是 Control 不是 HC
            seen[sid] = (grp, float(age))
        except: continue

ages = {}
for sid,(grp,age) in seen.items():
    ages.setdefault(grp,[]).append(age)

order = ['Prodromal','PD','HC','SWEDD']
colors = ['#F4A261','#E76F51','#2A9D8F','#E9C46A']
data = [ages.get(k,[]) for k in order]

fig,ax = plt.subplots(figsize=(5.8,5))
bp = ax.boxplot(data, patch_artist=True, widths=0.4, showfliers=True,
                flierprops={'marker':'o','markersize':3,'markerfacecolor':'gray','alpha':0.4})
for p,c in zip(bp['boxes'],colors): p.set_facecolor(c)
for m in bp['medians']: m.set_color('white')

ax.set_xticks([1,2,3,4])
ax.set_xticklabels([])
ax.set_ylabel('Age (years)', fontsize=12, fontweight='bold', labelpad=8)
ax.set_title('PPMI - Age Distribution by Diagnosis', fontsize=13, fontweight='bold', pad=16)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)
ymax = max(max(d) for d in data if d)
ax.set_ylim(0, ymax * 1.15)

y_offset = -0.05
trans = ax.get_xaxis_transform()
for i, (label, color) in enumerate(zip(order, colors)):
    ax.text(i + 1, y_offset, label, ha='center', va='top',
            fontsize=10, fontweight='bold', color=color, transform=trans)

leg = [mpatches.Patch(color=c,label='%s (n=%d)'%(o,len(ages.get(o,[])))) for o,c in zip(order,colors)]
ax.legend(handles=leg, loc='upper right', frameon=False, fontsize=8, handlelength=1.2)

fig.tight_layout(pad=1.5)
out = os.path.join(base, 'PPMI_age_boxplot.png')
plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.15); plt.close()
print('PPMI_age_boxplot.png saved.')