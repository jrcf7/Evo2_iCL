import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASETS = [
    ('miRNA',                      'miRNA'),
    ('Toxins_mxd',                 'Toxins'),
    ('AMR',                        'AMR'),
    ('tata_n5000_min20_max100',    'TATA Box 20-100'),
    ('tata_n5000_min1000_max3000', 'TATA Box 1k-3k'),
]

MODEL_MAP = {
    'evo2_20b': 'Evo2 20B',
    'evo2_40b': 'Evo2 40B',
}

ORDER = ['Evo2 20B', 'Evo2 40B']

# Evo2 40B: same Paired slot, marker, dash as evo2_mirna_allmodels (slot 1)
# Evo2 20B: Paired slot 9, marker pos 9, dash pos 9
_PAIRED = sns.color_palette('Paired', n_colors=10)
PALETTE = {
    'Evo2 20B': _PAIRED[9],   # purple
    'Evo2 40B': _PAIRED[1],   # dark blue
}
MARKERS = {
    'Evo2 20B': (6, 1, 30.0),
    'Evo2 40B': 'X',
}
DASHES = {
    'Evo2 20B': (3, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25),
    'Evo2 40B': (4, 1.5),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    rows = []
    for suffix, title in DATASETS:
        results_dir = f'../results_ppl_{suffix}/'
        if not os.path.isdir(results_dir):
            print(f'  WARNING: {results_dir} not found, skipping')
            continue
        for folder in os.listdir(results_dir):
            folder_path = os.path.join(results_dir, folder)
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
            df['dataset'] = title
            df['model']   = MODEL_MAP[model_raw]
            df['N']       = n_shot
            rows.append(df)

    if not rows:
        raise RuntimeError('No predictions found.')
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    print(f'  Loaded {len(predictions_df):,} rows  |  models: {sorted(predictions_df.model.unique())}')

    # 2. Macro F1
    f1_rows = []
    for (dataset, model, n), grp in predictions_df.groupby(['dataset', 'model', 'N']):
        _, _, f1, _ = precision_recall_fscore_support(
            grp['true_label'], grp['pred_label'],
            average='macro', zero_division=0
        )
        f1_rows.append({'dataset': dataset, 'model': model, 'N': n, 'f1': f1})
    f1_df = pd.DataFrame(f1_rows)

    # 3. Perplexity long format
    ppl_df = predictions_df.melt(
        id_vars=['dataset', 'model', 'N'],
        value_vars=['ppl_class0', 'ppl_class1'],
        var_name='scored_class',
        value_name='perplexity',
    )

    # 4. Shared row y-limits from group means
    f1_ylim  = row_ylim(f1_df['f1'])
    ppl_means = ppl_df.groupby(['dataset', 'model', 'N'])['perplexity'].mean()
    ppl_ylim  = row_ylim(ppl_means)
    y_formatter = ticker.FuncFormatter(fmt_3sig)

    # 5. Figure
    sns.set_theme(style='whitegrid', context='poster', font_scale=1.0)
    tick_fontsize  = 13
    label_fontsize = 15

    # Keep only datasets that have BOTH models
    both_present = [
        (suffix, title) for suffix, title in DATASETS
        if set(ORDER).issubset(
            set(predictions_df[predictions_df['dataset'] == title]['model'].unique())
        )
    ]
    if not both_present:
        raise RuntimeError('No datasets have both models.')
    dataset_titles = [t for _, t in both_present]
    n_cols = len(dataset_titles)
    print(f'  Datasets with both models: {dataset_titles}')

    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3.2, 6.5),
                             constrained_layout=True, sharex='col')
    fig.get_layout_engine().set(hspace=0.02, wspace=0.06, h_pad=0.15, w_pad=0.15)
    # Ensure axes is always 2-D even when n_cols == 1
    import numpy as np
    axes = np.atleast_2d(axes).reshape(2, n_cols)

    legend_ax = None  # will capture handles from first subplot

    def _lineplot(ax, data, y_col, capture_legend=False):
        if data.empty:
            return
        sns.lineplot(
            data=data, x='N', y=y_col,
            hue='model', style='model',
            hue_order=ORDER, style_order=ORDER,
            palette=PALETTE,
            markers=MARKERS, dashes=DASHES,
            linewidth=2.5,
            markersize=10,
            errorbar='ci',
            ax=ax, legend=capture_legend,
        )

    for col, title in enumerate(dataset_titles):
        ax_f1  = axes[0, col]
        ax_ppl = axes[1, col]

        # --- Row 0: Macro F1 ---
        # Capture legend on the very first subplot so handles include real styles
        capture = (col == 0)
        _lineplot(ax_f1, f1_df[f1_df['dataset'] == title], 'f1', capture_legend=capture)
        if capture:
            legend_ax = ax_f1
        ax_f1.set_ylim(*f1_ylim)
        ax_f1.set_title(title, fontsize=label_fontsize, fontweight='bold')
        ax_f1.set_xlabel('')
        ax_f1.set_ylabel('Macro F1' if col == 0 else '', fontsize=label_fontsize)
        ax_f1.tick_params(axis='x', labelbottom=True)
        style_ax(ax_f1, tick_fontsize)

        # --- Row 1: Perplexity ---
        _lineplot(ax_ppl, ppl_df[ppl_df['dataset'] == title], 'perplexity')
        ax_ppl.set_ylim(*ppl_ylim)
        ax_ppl.yaxis.set_major_formatter(y_formatter)
        ax_ppl.set_xlabel('N (per class)', fontsize=tick_fontsize)
        ax_ppl.set_ylabel('Perplexity' if col == 0 else '', fontsize=label_fontsize)
        style_ax(ax_ppl, tick_fontsize)

        # Set regular x-ticks AFTER both plots (sharex='col' propagates to row 0)
        ax_ppl.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))

    # 6. Shared legend — extract from the seaborn-rendered axis so line style,
    #    dash pattern, and marker are taken directly from the actual plotted handles.
    if legend_ax is not None:
        leg = legend_ax.get_legend()
        handles = leg.legend_handles
        labels  = [t.get_text() for t in leg.get_texts()]
        leg.remove()
    else:
        handles, labels = [], []

    fig.legend(
        handles, labels,
        loc='outside lower center',
        ncol=len(ORDER),
        frameon=True,
        fontsize=15,
        title=None,
    )

    out_path = 'evo2_size_comparison.png'
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    print(f'Saved {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
