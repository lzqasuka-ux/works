import csv, os, matplotlib.pyplot as plt, numpy as np
base = os.path.dirname(__file__)

def safe_read(fp):
    for enc in ['utf-8','gbk','latin-1','cp1252']:
        try:
            with open(fp,'r',encoding=enc) as f: return list(csv.DictReader(f)),enc
        except: continue
    return [],''

rows1,_ = safe_read(os.path.join(base,'ABIDE-1','Phenotypic_V1_0b.csv'))
rows2,_ = safe_read(os.path.join(base,'ABIDE-2','ABIDEII_Composite_Phenotypic.csv'))

counts = {}
for r in rows1:
    dx = r.get('DX_GROUP',''); sx = r.get('SEX','')
    label = 'ASD' if dx=='1' else 'HC' if dx=='2' else None
    if label:
        counts.setdefault(label, {'M':0,'F':0})
        counts[label]['M' if sx=='1' else 'F'] += 1

for r in rows2:
    dx = r.get('DX_GROUP',''); sx = r.get('SEX','').strip()
    label = 'ASD' if dx=='1' else 'HC' if dx=='2' else None
    if label:
        counts.setdefault(label, {'M':0,'F':0})
        counts[label]['M' if sx=='1' else 'F'] += 1

categories = ['ASD','HC']
male = [counts[c]['M'] for c in categories]
female = [counts[c]['F'] for c in categories]

x = np.arange(len(categories)); w = 0.5
fig,ax = plt.subplots(figsize=(5.5,5))
b1 = ax.bar(x, male, w, label='Male', color='#4A90D9', edgecolor='white')
b2 = ax.bar(x, female, w, bottom=male, label='Female', color='#E88C8C', edgecolor='white')

for bar, m, f in zip(x, male, female):
    if m > 0: ax.text(bar, m/2, str(m), ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    if f > 0: ax.text(bar, m + f/2, str(f), ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(bar, m + f + 15, str(m+f), ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333')

ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=12)
ax.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold', labelpad=8)
ax.set_title('ABIDE - Sex Distribution by Diagnosis', fontsize=13, fontweight='bold', pad=16)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.04), frameon=False, fontsize=10)
ax.set_ylim(0, max(m+f for m,f in zip(male,female)) * 1.22)
ax.tick_params(labelsize=11); ax.yaxis.set_major_locator(plt.MaxNLocator(6))

fig.tight_layout(pad=1.5)
plt.savefig(os.path.join(base,'ABIDE_sex_stacked.png'), dpi=300, bbox_inches='tight', pad_inches=0.15)
plt.close()
print('ABIDE_sex_stacked.png saved.')