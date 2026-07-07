import csv, os, matplotlib.pyplot as plt, numpy as np
base = os.path.dirname(__file__)

def safe_read(fp):
    for enc in ['utf-8','gbk','latin-1','cp1252']:
        try:
            with open(fp,'r',encoding=enc) as f: return list(csv.DictReader(f)),enc
        except: continue
    return [],''

rows,_ = safe_read(os.path.join(base,'allSubs_testSet_phenotypic_dx.csv'))
dx_map = {'0':'TD','1':'ADHD-C','2':'ADHD-HI','3':'ADHD-I'}
counts = {}
for r in rows:
    dx = r.get('DX','').strip()
    gd = r.get('Gender','').strip()
    label = dx_map.get(dx)
    if not label: continue
    counts.setdefault(label, {'M':0,'F':0})
    counts[label]['M' if gd=='1' else 'F'] += 1

order = ['TD','ADHD-C','ADHD-I','ADHD-HI']
male = [counts[c]['M'] for c in order]
female = [counts[c]['F'] for c in order]

x = np.arange(len(order)); w = 0.5
fig,ax = plt.subplots(figsize=(6.5,5))
b1 = ax.bar(x, male, w, label='Male', color='#4A90D9', edgecolor='white')
b2 = ax.bar(x, female, w, bottom=male, label='Female', color='#E88C8C', edgecolor='white')

for i,(m,f) in enumerate(zip(male,female)):
    if m>0: ax.text(i, m/2, str(m), ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    if f>0: ax.text(i, m+f/2, str(f), ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(i, m+f+2, str(m+f), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333')

ax.set_xticks(x); ax.set_xticklabels(order, fontsize=10)
ax.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold', labelpad=8)
ax.set_title('ADHD-200 - Sex Distribution by Diagnosis', fontsize=13, fontweight='bold', pad=16)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.04), frameon=False, fontsize=10)
ax.set_ylim(0, max(m+f for m,f in zip(male,female)) * 1.28)
ax.tick_params(labelsize=10); ax.yaxis.set_major_locator(plt.MaxNLocator(6))

fig.tight_layout(pad=1.5)
plt.savefig(os.path.join(base,'AHD200_sex_stacked.png'), dpi=300, bbox_inches='tight', pad_inches=0.15)
plt.close()
print('AHD200_sex_stacked.png saved.')