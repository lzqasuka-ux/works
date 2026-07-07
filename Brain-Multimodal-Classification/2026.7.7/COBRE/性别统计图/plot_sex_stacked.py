import csv, os, matplotlib.pyplot as plt, numpy as np
base = os.path.dirname(__file__)

def safe_read(fp):
    for enc in ['utf-8','gbk','latin-1','cp1252']:
        try:
            with open(fp,'r',encoding=enc) as f: return list(csv.DictReader(f)),enc
        except: continue
    return [],[]

rows,_ = safe_read(os.path.join(base,'COBRE_phenotypic_data.csv'))
counts = {}
for r in rows:
    st = r.get('Subject Type',''); gd = r.get('Gender',''); dx = r.get('Diagnosis','')
    if st == 'Control':
        lbl = 'HC'
    elif st == 'Patient' and dx and dx != 'None':
        lbl = 'SZ'
    else:
        continue
    counts.setdefault(lbl, {'M':0,'F':0})
    counts[lbl]['M' if gd=='Male' else 'F'] += 1

order = ['SZ','HC']
male = [counts[c]['M'] for c in order]
female = [counts[c]['F'] for c in order]

x = np.arange(len(order)); w = 0.45
fig,ax = plt.subplots(figsize=(5,5))
b1 = ax.bar(x, male, w, label='Male', color='#4A90D9', edgecolor='white')
b2 = ax.bar(x, female, w, bottom=male, label='Female', color='#E88C8C', edgecolor='white')

for i,(m,f) in enumerate(zip(male,female)):
    if m>0: ax.text(i, m/2, str(m), ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    if f>0: ax.text(i, m+f/2, str(f), ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(i, m+f+1.5, str(m+f), ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333')

ax.set_xticks(x); ax.set_xticklabels(order, fontsize=12)
ax.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold', labelpad=8)
ax.set_title('COBRE - Sex Distribution by Diagnosis', fontsize=13, fontweight='bold', pad=16)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.04), frameon=False, fontsize=10)
ax.set_ylim(0, max(m+f for m,f in zip(male,female)) * 1.35)
ax.tick_params(labelsize=11); ax.yaxis.set_major_locator(plt.MaxNLocator(5))

fig.tight_layout(pad=1.5)
plt.savefig(os.path.join(base,'COBRE_sex_stacked.png'), dpi=300, bbox_inches='tight', pad_inches=0.15)
plt.close()
print('COBRE_sex_stacked.png saved.')
