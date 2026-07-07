import csv, os, matplotlib.pyplot as plt, matplotlib.patches as mpatches
base = os.path.dirname(__file__)

def safe_read(fp):
    for enc in ['utf-8','gbk','latin-1','cp1252']:
        try:
            with open(fp,'r',encoding=enc) as f: return list(csv.DictReader(f)),enc
        except: continue
    with open(fp,'r',encoding='utf-8',errors='replace') as f: return list(csv.DictReader(f)),'replace'

rows,_ = safe_read(os.path.join(base,'allSubs_testSet_phenotypic_dx.csv'))
ages = {}
for row in rows:
    grp = row.get('DX',''); age = row.get('Age','')
    try: age = float(age)
    except: continue
    if grp: ages.setdefault(grp,[]).append(age)

order = ['Typically Developing','ADHD-Combined','ADHD-Inattentive','ADHD-Hyperactive/Impulsive']
labels = ['TD','ADHD-C','ADHD-I','ADHD-HI']
colors = ['#2A9D8F','#E76F51','#F4A261','#E9C46A']
data = [ages.get(k,[]) for k in order]

fig,ax = plt.subplots(figsize=(6,5))
bp = ax.boxplot(data, patch_artist=True, widths=0.45, showfliers=True,
                flierprops={'marker':'o','markersize':3,'markerfacecolor':'gray','alpha':0.4})
for p,c in zip(bp['boxes'],colors): p.set_facecolor(c)
for m in bp['medians']: m.set_color('white')

ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Age (years)', fontsize=12, fontweight='bold', labelpad=8)
ax.set_title('ADHD-200 - Age Distribution by Diagnosis', fontsize=13, fontweight='bold', pad=16)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)
ymax = max(max(d) for d in data if d)
ax.set_ylim(0, ymax * 1.12)

leg = [mpatches.Patch(color='#2A9D8F',label='TD (n=%d)'%len(ages.get(order[0],[]))),
       mpatches.Patch(color='#E76F51',label='ADHD-C (n=%d)'%len(ages.get(order[1],[]))),
       mpatches.Patch(color='#F4A261',label='ADHD-I (n=%d)'%len(ages.get(order[2],[]))),
       mpatches.Patch(color='#E9C46A',label='ADHD-HI (n=%d)'%len(ages.get(order[3],[])))]
ax.legend(handles=leg, loc='upper right', frameon=False, fontsize=8.5, handlelength=1.2)

fig.tight_layout(pad=1.5)
out = os.path.join(base, 'AHD200_age_boxplot.png')
plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.15); plt.close()
print('AHD200_age_boxplot.png saved.')
