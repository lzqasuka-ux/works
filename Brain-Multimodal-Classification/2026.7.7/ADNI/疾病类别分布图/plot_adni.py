import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'

categories = ['CN', 'MCI', 'AD', 'EMCI', 'SMC', 'LMCI']
counts     = [589, 347, 87, 64, 34, 29]
colors     = ['#2A9D8F', '#E9C46A', '#E76F51', '#F4A261', '#A8DADC', '#457B9D']

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(categories, counts, color=colors, width=0.55, edgecolor='white', linewidth=0.8)

for bar, v in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(v), ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333')

ax.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
ax.set_title('ADNI Dataset Distribution', fontsize=13, fontweight='bold', pad=10)
ax.set_ylim(0, max(counts) * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

plt.tight_layout()
plt.savefig(r'C:\Users\zhongqing.lu\Desktop\works\数据集\ADNI\ADNI_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print('ADNI_distribution.png saved.')