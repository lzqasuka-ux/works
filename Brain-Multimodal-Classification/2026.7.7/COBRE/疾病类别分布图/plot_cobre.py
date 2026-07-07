import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'

categories = ['SZ', 'HC']
counts     = [72, 74]
colors     = ['#E76F51', '#2A9D8F']

fig, ax = plt.subplots(figsize=(4.5, 4))
bars = ax.bar(categories, counts, color=colors, width=0.45, edgecolor='white', linewidth=0.8)

for bar, v in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            str(v), ha='center', va='bottom', fontsize=13, fontweight='bold', color='#333')

ax.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
ax.set_title('COBRE Dataset Distribution', fontsize=13, fontweight='bold', pad=10)
ax.set_ylim(0, max(counts) * 1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=11)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

plt.tight_layout()
plt.savefig(r'C:\Users\zhongqing.lu\Desktop\works\数据集\COBRE\COBRE_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print('COBRE_distribution.png saved.')