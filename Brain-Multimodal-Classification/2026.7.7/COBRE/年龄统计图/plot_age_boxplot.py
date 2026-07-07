import csv, os, matplotlib.pyplot as plt, matplotlib.patches as mpatches
base = os.path.dirname(__file__)

def safe_read(fp):
    for enc in ['utf-8','gbk','latin-1','cp1252']:
        try:
            with open(fp,'r',encoding=enc) as f: return list(csv.DictReader(f)),enc
        except: continue
    with open(fp,'r',encoding='utf-8',errors='replace') as f: return list(csv.DictReader(f)),'replace'

rows,_ = safe_read(os.path.join(base,'COBRE_phenotypic_data.csv'))
ages = {}
for row in rows:
    st = row.get('Subject Type',''); age = row.get('Current Age',''); dx = row.get('Diagnosis','')
    try: age = float(age)
    except: continue
    if st == 'Control': ages.setdefault('HC',[]).append(age)
    elif st == 'Patient' and dx and dx != 'None': ages.setdefault('SZ',[]).append(age)

order = ['SZ','HC']; colors = ['#E76F51','#2A9D8F']
data = [ages.get(k,[]) for k in order]

fig,ax = plt.subplots(figsize=(4.8,5))
bp = ax.boxplot(data, patch_artist=True, widths=0.4, showfliers=True,
                flierprops={'marker':'o','markersize':4,'markerfacecolor':'gray','alpha':0.5})
for p,c in zip(bp['boxes'],colors): p.set_facecolor(c)
for m in bp['medians']: m.set_color('white')

ax.set_xticks([1, 2])
ax.set_xticklabels([])
ax.set_ylabel('Age (years)', fontsize=12, fontweight='bold', labelpad=8)
ax.set_title('COBRE - Age Distribution by Diagnosis', fontsize=13, fontweight='bold', pad=16)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=11)
ymax = max(max(d) for d in data if d)
ax.set_ylim(0, ymax * 1.15)

y_offset = -0.05   # 改这个：-0.03=靠近柱子, -0.08=更远
trans = ax.get_xaxis_transform()
for i, (label, color) in enumerate(zip(order, colors)):
    ax.text(i + 1, y_offset, label, ha='center', va='top',
            fontsize=12, fontweight='bold', color=color,
            transform=trans)

leg = [mpatches.Patch(color='#E76F51',label='SZ (n=%d)'%len(ages.get('SZ',[]))),
       mpatches.Patch(color='#2A9D8F',label='HC (n=%d)'%len(ages.get('HC',[])))]
ax.legend(handles=leg, loc='upper right', frameon=False, fontsize=10, handlelength=1.2)

fig.tight_layout(pad=1.5)
out = os.path.join(base, 'COBRE_age_boxplot.png')
plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.15); plt.close()
print('COBRE_age_boxplot.png saved.')
