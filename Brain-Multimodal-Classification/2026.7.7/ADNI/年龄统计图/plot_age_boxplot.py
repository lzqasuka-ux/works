import csv, os, matplotlib.pyplot as plt, matplotlib.patches as mpatches
base = os.path.dirname(__file__)

def safe_read(fp):
    for enc in ['utf-8','gbk','latin-1','cp1252']:
        try:
            with open(fp,'r',encoding=enc) as f: return list(csv.DictReader(f)),enc
        except: continue
    with open(fp,'r',encoding='utf-8',errors='replace') as f: return list(csv.DictReader(f)),'replace'

rows,_ = safe_read(os.path.join(base,'ADNI-3','ADNI_7_06_2026.csv'))
seen = {}
for row in rows:
    sid = row.get('Subject',''); grp = row.get('Group',''); age = row.get('Age','')
    if sid and sid not in seen:
        try: seen[sid] = (grp, float(age))
        except: continue

ages = {}
for sid,(grp,age) in seen.items(): ages.setdefault(grp,[]).append(age)

order = ['CN','MCI','EMCI','LMCI','SMC','AD']
colors = ['#2A9D8F','#F4A261','#F4A261','#F4A261','#F4A261','#E76F51']
data = [ages.get(k,[]) for k in order]

fig,ax = plt.subplots(figsize=(7.5,5))
bp = ax.boxplot(data, patch_artist=True, widths=0.45, showfliers=True,
                flierprops={'marker':'o','markersize':3,'markerfacecolor':'gray','alpha':0.4})
for p,cl in zip(bp['boxes'],colors): p.set_facecolor(cl)
for m in bp['medians']: m.set_color('white')

ax.set_xticklabels(order, fontsize=10)
ax.set_ylabel('Age (years)', fontsize=12, fontweight='bold', labelpad=8)
ax.set_title('ADNI - Age Distribution by Diagnosis', fontsize=13, fontweight='bold', pad=16)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)
ymax = max(max(d) for d in data if d)
ax.set_ylim(0, ymax * 1.12)

mci_n = sum(len(ages.get(k,[])) for k in ['MCI','EMCI','LMCI','SMC'])
leg = [mpatches.Patch(color='#2A9D8F',label='CN (n=%d)'%len(ages.get('CN',[]))),
       mpatches.Patch(color='#F4A261',label='MCI Spectrum (n=%d)'%mci_n),
       mpatches.Patch(color='#E76F51',label='AD (n=%d)'%len(ages.get('AD',[])))]
ax.legend(handles=leg, loc='upper right', frameon=False, fontsize=9, handlelength=1.2)

fig.tight_layout(pad=1.5)
out = os.path.join(base, 'ADNI_age_boxplot.png')
plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.15); plt.close()
print('ADNI_age_boxplot.png saved.')
