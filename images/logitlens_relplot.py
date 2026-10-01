import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Config — same palette/markers/dashes as evo2_mirna_allmodels.py
# ---------------------------------------------------------------------------

ORDER = [
    'Evo2 7B',
    'Evo2 40B',
    'Evo2 7B Base',
    'Evo2 40B Base',
    'Evo2 1B Base',
]

SIZE_GROUP = {
    'Evo2 40B':      '40B',
    'Evo2 40B Base': '40B',
    'Evo2 7B':       '7B',
    'Evo2 7B Base':  '7B',
    'Evo2 1B Base':  '1B',
}
SIZE_ORDER = ['40B', '7B', '1B']

_PAIRED = sns.color_palette('Paired', n_colors=7)
_SLOTS  = [0, 1, 4, 5, 6]
PALETTE = {m: _PAIRED[_SLOTS[i]] for i, m in enumerate(ORDER)}

MARKERS = dict(zip(ORDER, ['o', 'X', (4, 0, 0), (4, 1, 0), '^']))
DASHES  = dict(zip(ORDER, [
    '',
    (4, 1.5),
    (5, 1, 1, 1),
    (3, 1.25, 1.25, 1.25, 1.25, 1.25),
    (4, 1, 4, 1, 1, 1),
]))

CSV_PATH = 'logitlens_models.csv'
OUT_PATH = 'logitlens_relplot.png'

# ---------------------------------------------------------------------------
# Load & reshape
# ---------------------------------------------------------------------------

with open(CSV_PATH, newline='') as fh:
    rows = list(csv.reader(fh))
n_cols = max(len(r) for r in rows)
header = rows[0] + [f'L{i}' for i in range(len(rows[0]) - 1, n_cols - 1)]
data   = [r + [''] * (n_cols - len(r)) for r in rows[1:]]
df_wide = pd.DataFrame(data, columns=header).set_index(header[0])

df = (df_wide
      .reset_index()
      .melt(id_vars='Model', var_name='layer_str', value_name='val')
      .dropna(subset=['val'])
      .query("val != ''"))

df[['mean', 'std']] = df['val'].str.split('±', expand=True).astype(float)
df['layer'] = df['layer_str'].str.lstrip('L').astype(int)
df = df[df['Model'].isin(ORDER)]
df['size'] = df['Model'].map(SIZE_GROUP)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

sns.set_theme(style='whitegrid', context='poster', font_scale=1.0)

g = sns.relplot(
    data=df,
    x='layer', y='mean',
    col='size', col_order=SIZE_ORDER,
    hue='Model', hue_order=ORDER,
    style='Model', style_order=ORDER,
    kind='line',
    palette=PALETTE,
    markers=MARKERS,
    dashes=DASHES,
    linewidth=2.5,
    markersize=8,
    errorbar=None,
    facet_kws={'sharey': True, 'sharex': False},
    height=4, aspect=1.2,
    legend=False,
)

# Shaded std bands, titles, labels, grid
size_models = {'40B': ['Evo2 40B', 'Evo2 40B Base'],
               '7B':  ['Evo2 7B',  'Evo2 7B Base'],
               '1B':  ['Evo2 1B Base']}

for size, ax in zip(SIZE_ORDER, g.axes.flat):
    for model in size_models[size]:
        sub = df[df['Model'] == model].sort_values('layer')
        ax.fill_between(sub['layer'],
                        sub['mean'] - sub['std'],
                        sub['mean'] + sub['std'],
                        color=PALETTE[model], alpha=0.15)
    ax.set_title(f'Evo2 {size}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('Mean P(top-1 token)' if size == SIZE_ORDER[0] else '', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(axis='both', labelsize=11)

# Build legend handles manually
from matplotlib.lines import Line2D as _Line2D
handles, labels = [], []
for model in ORDER:
    dash = DASHES[model]
    ls = '-' if dash == '' else (0, dash)
    handles.append(_Line2D([0], [0],
                           color=PALETTE[model],
                           marker=MARKERS[model],
                           linestyle=ls,
                           linewidth=2.5, markersize=8,
                           label=model))
    labels.append(model)

g.figure.legend(
    handles, labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.08),
    ncol=5,
    frameon=True,
    fontsize=12,
    title=None,
)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f'Saved to {OUT_PATH}')
