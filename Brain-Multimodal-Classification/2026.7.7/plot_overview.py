import matplotlib.pyplot as plt
import numpy as np

datasets = ['ABIDE', 'ADNI', 'AHD200', 'COBRE', 'PPMI']
subjects = [2226, 1150, 197, 148, 1424]
smri     = [2226, 31828, 197, 148, 20966]

x = np.arange(len(datasets))
w = 0.28  # 细柱宽度

fig, ax = plt.subplots(figsize=(9, 5.5))

bars1 = ax.bar(x - w/2, subjects, w, label='Subjects', color='#4A90D9', edgecolor='white')
bars2 = ax.bar(x + w/2, smri, w, label='sMRI Scans', color='#E76F51', edgecolor='white')

# Annotate values on top of bars
for bar, v in zip(bars1, subjects):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(subjects)*0.01,
            str(v), ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#4A90D9')
for bar, v in zip(bars2, smri):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(smri)*0.01,
            str(v), ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#E76F51')

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylabel('Count', fontsize=12, fontweight='bold', labelpad=8)
ax.set_title('Five Datasets Overview – Subjects & sMRI Scans', fontsize=13, fontweight='bold', pad=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', frameon=False, fontsize=10)
ax.tick_params(labelsize=10)
ax.yaxis.set_major_locator(plt.MaxNLocator(6))

fig.tight_layout(pad=1.5)
plt.savefig('C:/Users/zhongqing.lu/Desktop/works/数据集/Five_Datasets_Overview.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
plt.close()
print('Five_Datasets_Overview.png saved.')