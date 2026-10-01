import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_MAP = {
    'evo2_1b_base':  'Evo2 1B Base',
    'evo2_7b':       'Evo2 7B',
    'evo2_7b_base':  'Evo2 7B Base',
    'evo2_40b':      'Evo2 40B',
    'evo2_40b_base': 'Evo2 40B Base',
}

ORDER = [
    'Evo2 7B',
    'Evo2 40B',
    'Evo2 7B Base',
    'Evo2 40B Base',
    'Evo2 1B Base',
]

# Skip palette slots 2 and 3; use Paired indices 0,1,4,5,6
_PAIRED  = sns.color_palette('Paired', n_colors=7)
_SLOTS   = [0, 1, 4, 5, 6]
PALETTE  = {m: _PAIRED[_SLOTS[i]] for i, m in enumerate(ORDER)}

# Skip seaborn's default marker/dash slots 2 and 3
MARKERS = dict(zip(ORDER, ['o', 'X', (4, 0, 0), (4, 1, 0), '^']))
DASHES  = dict(zip(ORDER, [
    '',
    (4, 1.5),
    (5, 1, 1, 1),
    (3, 1.25, 1.25, 1.25, 1.25, 1.25),
    (4, 1, 4, 1, 1, 1),
]))

RESULTS_DIR = '../results_ppl_miRNA/'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    rows = []
    for folder in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, folder)
        if not os.path.isdir(folder_path) or 'DEP' in folder_path:
            continue
        try:
            n_shot    = int(folder.split('_')[-1].replace('shot', ''))
            model_raw = '_'.join(folder.split('_')[:-1])
        except ValueError:
            continue
        if model_raw not in MODEL_MAP:
            continue
        pred_path = os.path.join(folder_path, 'predictions.tsv')
        if not os.path.exists(pred_path):
            continue
        df = pd.read_csv(pred_path, sep='\t')
        df['model'] = MODEL_MAP[model_raw]
        df['N']     = n_shot
        rows.append(df)

    if not rows:
        raise RuntimeError(f'No predictions found in {RESULTS_DIR}')
    return pd.concat(rows, ignore_index=True)


def row_ylim(series, margin=0.08):
    lo, hi = series.min(), series.max()
    pad = (hi - lo) * margin if hi > lo else abs(hi) * margin or margin
    return lo - pad, hi + pad


def fmt_3sig(x, _):
    if abs(x) >= 1_000_000:
        v, s = x / 1_000_000, 'M'
    elif abs(x) >= 1_000:
        v, s = x / 1_000, 'k'
    else:
        v, s = x, ''
    t = f'{v:.{2 if s else 3}g}'
    if '.' not in t and abs(v) < 100:
        t += '.0'
    return t + s


def style_ax(ax, tick_fontsize):
    ax.grid(True, linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(axis='both', labelsize=tick_fontsize)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load
    predictions_df = load_data()
    print(f'  Loaded {len(predictions_df):,} rows')
    print(f'  Models: {sorted(predictions_df.model.unique())}')

    # Filter to N ≤ 32
    predictions_df = predictions_df[predictions_df['N'] <= 32]

    # 2. Macro F1
    f1_rows = []
    for (model, n), grp in predictions_df.groupby(['model', 'N']):
        _, _, f1, _ = precision_recall_fscore_support(
            grp['true_label'], grp['pred_label'],
            average='macro', zero_division=0
        )
        f1_rows.append({'model': model, 'N': n, 'f1': f1})
    f1_df = pd.DataFrame(f1_rows)

    # 3. Y-limits from group means
    f1_ylim = row_ylim(f1_df['f1'])

    # 4. Figure — single panel
    sns.set_theme(style='whitegrid', context='poster', font_scale=1.0)
    tick_fontsize  = 13
    label_fontsize = 15

    fig, ax_f1 = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
    fig.suptitle('miRNA — Evo2 Models', fontsize=17, fontweight='bold')

    sns.lineplot(
        data=f1_df, x='N', y='f1',
        hue='model', style='model',
        hue_order=ORDER, style_order=ORDER,
        palette=PALETTE,
        markers=MARKERS, dashes=DASHES,
        linewidth=2.5, markersize=10,
        errorbar='ci',
        ax=ax_f1, legend=True,
    )
    ax_f1.set_ylim(*f1_ylim)
    ax_f1.set_ylabel('Macro F1', fontsize=label_fontsize)
    ax_f1.set_xlabel('N (per class)', fontsize=tick_fontsize)
    ax_f1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
    style_ax(ax_f1, tick_fontsize)

    # 6. Steal legend from seaborn (preserves real line style + marker)
    leg = ax_f1.get_legend()
    handles = leg.legend_handles
    labels  = [t.get_text() for t in leg.get_texts()]
    leg.remove()

    ax_f1.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=True,
        fontsize=13,
        title=None,
    )

    out_path = 'evo2_mirna_allmodels.png'
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    print(f'Saved {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
