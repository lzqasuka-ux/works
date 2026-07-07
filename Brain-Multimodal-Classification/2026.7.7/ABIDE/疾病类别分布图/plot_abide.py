import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'

# Data
categories = ['ASD', 'HC']
counts     = [1060, 1166]
colors     = ['#E76F51', '#2A9D8F']

fig, ax = plt.subplots(figsize=(5, 4.5))
bars = ax.bar(categories, counts, color=colors, width=0.5, edgecolor='white', linewidth=0.8)

# Value labels on top of each bar
for bar, v in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            str(v), ha='center', va='bottom', fontsize=13, fontweight='bold', color='#333333')

ax.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
ax.set_title('ABIDE Dataset Distribution', fontsize=13, fontweight='bold', pad=10)
ax.set_ylim(0, max(counts) * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=11)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

plt.tight_layout()
plt.savefig(r'C:\Users\zhongqing.lu\Desktop\works\数据集\ABIDE\ABIDE_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print('ABIDE_distribution.png saved.')