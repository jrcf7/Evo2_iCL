import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.metrics import precision_recall_fscore_support

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASETS = [
    # (results_dir_suffix,           meta_prefix,                  display_title)
    ('miRNA',                        'mrna',                       'miRNA'),
    ('Toxins_mxd',                   'toxinmix',                   'Toxins'),
    ('AMR',                          'amr',                        'AMR'),
    ('tata_n5000_min20_max100',      'tata_n5000_min20_max100',    'TATA Box 20-100'),
    ('tata_n5000_min1000_max3000',   'tata_n5000_min1000_max3000', 'TATA Box 1k-3k'),
]

MODEL_NAME_MAP = {
    'evo2_1b_base':         'Evo2 1B Base',
    'evo2_7b':              'Evo2 7B',
    'evo2_20b':             'Evo2 20B',
    'evo2_40b':             'Evo2 40B',
    'evo2_7b_base':         'Evo2 7B Base',
    'evo2_40b_base':        'Evo2 40B Base',
    'medgemma-1.5-4b-it':   'MedGemma 1.5 4B',
    'medgemma-27b-text-it': 'MedGemma txt 27B',
    'gemma-4-31B-it':       'Gemma 4 31B IT',
}

ORDER = [
    'Evo2 7B', 'Evo2 40B',
    'MedGemma 1.5 4B', 'MedGemma txt 27B',
]

# Palette built from seaborn Paired, keyed per model in ORDER.
_PAIRED = sns.color_palette('Paired', n_colors=len(ORDER))
MODEL_PALETTE = {m: _PAIRED[i] for i, m in enumerate(ORDER)}

# Models to exclude from all plots (display names).
DISCARD_MODELS = [
    'Evo2 20B',
    'Gemma 4 31B IT',
]

# Row 1 = Evo2 PPL (linear, low scale ~1-3)
# Row 2 = LLM PPL  (log scale, wide range)
EVO_TYPES = {'Evo 2 1M'}
LLM_TYPES = {'MedGemma', 'Gemma 4'}

TYPE_ORDER = ['Evo 2 1M', 'MedGemma', 'Gemma 4']

_EVO2_BASE_RE = re.compile(r"evo2.*base", re.IGNORECASE)
_EVO2_RE      = re.compile(r"evo2",       re.IGNORECASE)


def get_model_type(model_raw: str) -> str:
    if _EVO2_BASE_RE.search(model_raw):
        return 'Evo 2 Base'
    if _EVO2_RE.search(model_raw):
        return 'Evo 2 1M'
    if 'med' in model_raw.lower():
        return 'MedGemma'
    return 'Gemma 4'


def filter_models(df: pd.DataFrame, discard: list = DISCARD_MODELS) -> pd.DataFrame:
    """Drop rows whose display model name is in the discard list."""
    return df[~df['model'].isin(discard)].copy()


def fmt_3sig(x, _):
    if abs(x) >= 1_000_000:
        v, suffix = x / 1_000_000, 'M'
    elif abs(x) >= 1000:
        v, suffix = x / 1000, 'k'
    else:
        v, suffix = x, ''
    precision = 2 if suffix else 3
    s = f'{v:.{precision}g}'
    if '.' not in s and abs(v) < 100:
        s += '.0'
    return s + suffix


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(suffix: str, title: str) -> pd.DataFrame:
    results_dir = f'../results_ppl_{suffix}/'
    rows = []
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path) or 'DEP' in folder_path:
            continue
        try:
            n_shot    = int(folder.split('_')[-1].replace('shot', ''))
            model_raw = '_'.join(folder.split('_')[:-1])
        except ValueError:
            continue

        pred_path = os.path.join(folder_path, 'predictions.tsv')
        if not os.path.exists(pred_path):
            continue

        df = pd.read_csv(pred_path, sep='\t')
        df['dataset'] = title
        df['model']   = MODEL_NAME_MAP.get(model_raw, model_raw)
        df['type']    = get_model_type(model_raw)
        df['N']       = n_shot
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load all datasets
    all_dfs = []
    for suffix, _meta_prefix, title in DATASETS:
        df = load_dataset(suffix, title)
        if not df.empty:
            all_dfs.append(df)
            print(f"  Loaded {title}: {len(df):,} rows")
        else:
            print(f"  WARNING: no predictions found for {title}")

    predictions_df = pd.concat(all_dfs, ignore_index=True)

    # 2. Exclude base models, then apply the discard list
    predictions_df = predictions_df[predictions_df['type'] != 'Evo 2 Base'].copy()
    predictions_df = filter_models(predictions_df)

    # 3. Compute Macro F1 per (dataset, model, type, N)
    f1_rows = []
    for (dataset, model, typ, n), grp in predictions_df.groupby(['dataset', 'model', 'type', 'N']):
        _, _, f1, _ = precision_recall_fscore_support(
            grp['true_label'], grp['pred_label'],
            average='macro', zero_division=0
        )
        f1_rows.append({'dataset': dataset, 'model': model, 'type': typ, 'N': n, 'f1': f1})
    f1_df = pd.DataFrame(f1_rows)

    # 4. Perplexity long format (keep type column)
    ppl_df = predictions_df.melt(
        id_vars=['dataset', 'model', 'type', 'N'],
        value_vars=['ppl_class0', 'ppl_class1'],
        var_name='scored_class',
        value_name='perplexity',
    )

    # 5. Build figure: 3 rows × 5 cols
    #    row 0 — Macro F1      (all models)
    #    row 1 — PPL Evo2      (linear, low scale ~1–3)
    #    row 2 — PPL LLMs      (log scale)
    sns.set_theme(style='whitegrid', context='poster', font_scale=1.0)
    tick_fontsize  = 17
    label_fontsize = 20
    title_fontsize = 22
    y_formatter    = FuncFormatter(fmt_3sig)

    fig, axes = plt.subplots(3, 5, figsize=(21, 11), constrained_layout=True,
                             sharex='col')
    fig.get_layout_engine().set(hspace=0.02, wspace=0.04, h_pad=0.1, w_pad=0.1)

    dataset_titles = [d[2] for d in DATASETS]

    def _style_ax(ax):
        ax.grid(True, linestyle='--', alpha=0.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(axis='both', labelsize=tick_fontsize)

    def _row_ylim(means, margin=0.08):
        """Compute shared y-limits from a series of group means."""
        lo, hi = means.min(), means.max()
        pad = (hi - lo) * margin if hi > lo else abs(hi) * margin or margin
        return lo - pad, hi + pad

    # Pre-compute global y-limits from GROUP MEANS (what seaborn actually plots),
    # not from raw per-sequence values which contain extreme outliers.
    f1_ylim = _row_ylim(f1_df['f1'])

    evo_means = (
        ppl_df[ppl_df['type'].isin(EVO_TYPES)]
        .groupby(['dataset', 'model', 'N'])['perplexity'].mean()
    )
    llm_means = (
        ppl_df[ppl_df['type'].isin(LLM_TYPES)]
        .groupby(['dataset', 'model', 'N'])['perplexity'].mean()
    )
    evo_ylim = _row_ylim(evo_means) if not evo_means.empty else (None, None)
    llm_ylim = _row_ylim(llm_means) if not llm_means.empty else (None, None)

    _legend_ax = [None]  # will hold the axis from which we steal seaborn handles

    def _lineplot(ax, data, y_col, present, capture_legend=False):
        if data.empty or not present:
            return
        sns.lineplot(
            data=data,
            x='N', y=y_col,
            hue='model', style='model',
            hue_order=present,
            style_order=ORDER,
            palette={m: MODEL_PALETTE[m] for m in present if m in MODEL_PALETTE},
            markers=True, dashes=True,
            errorbar='ci',
            ax=ax, legend=capture_legend,
        )
        if capture_legend:
            _legend_ax[0] = ax

    for col, title in enumerate(dataset_titles):
        ds_pred  = predictions_df[predictions_df['dataset'] == title]
        present  = [m for m in ORDER if m in ds_pred['model'].unique()]

        evo_present = [m for m in present if ds_pred[ds_pred['model'] == m]['type'].iloc[0] in EVO_TYPES]
        llm_present = [m for m in present if ds_pred[ds_pred['model'] == m]['type'].iloc[0] in LLM_TYPES]

        ax_f1      = axes[0, col]
        ax_ppl_evo = axes[1, col]
        ax_ppl_llm = axes[2, col]

        # --- Row 0: Macro F1 (all models) ---
        data_f1 = f1_df[f1_df['dataset'] == title]
        _lineplot(ax_f1, data_f1, 'f1', present, capture_legend=(col == 0))
        ax_f1.set_ylim(*f1_ylim)
        ax_f1.set_title(title, fontsize=title_fontsize, fontweight='bold')
        ax_f1.set_xlabel('')
        ax_f1.set_ylabel('Macro F1' if col == 0 else '', fontsize=label_fontsize)
        ax_f1.tick_params(axis='x', labelbottom=True)
        _style_ax(ax_f1)

        # --- Row 1: Evo2 PPL (linear) ---
        data_ppl_evo = ppl_df[(ppl_df['dataset'] == title) & (ppl_df['type'].isin(EVO_TYPES))]
        _lineplot(ax_ppl_evo, data_ppl_evo, 'perplexity', evo_present)
        if evo_ylim[0] is not None:
            ax_ppl_evo.set_ylim(*evo_ylim)
        ax_ppl_evo.yaxis.set_major_formatter(y_formatter)
        ax_ppl_evo.set_xlabel('')
        ax_ppl_evo.set_ylabel('Perplexity' if col == 0 else '', fontsize=label_fontsize)
        ax_ppl_evo.tick_params(axis='x', labelbottom=True)
        _style_ax(ax_ppl_evo)

        # --- Row 2: LLM PPL (linear) ---
        data_ppl_llm = ppl_df[(ppl_df['dataset'] == title) & (ppl_df['type'].isin(LLM_TYPES))]
        _lineplot(ax_ppl_llm, data_ppl_llm, 'perplexity', llm_present)
        if llm_ylim[0] is not None:
            ax_ppl_llm.set_ylim(*llm_ylim)
        ax_ppl_llm.yaxis.set_major_formatter(y_formatter)
        ax_ppl_llm.set_xlabel('N (per class)', fontsize=label_fontsize)
        ax_ppl_llm.set_ylabel('Perplexity' if col == 0 else '', fontsize=label_fontsize)
        _style_ax(ax_ppl_llm)

        # Apply regular tick intervals after all plots in this column are drawn.
        # sharex='col' propagates these ticks to rows 0 and 1 automatically.
        ax_ppl_llm.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))


    # 6. Shared legend — steal handles from seaborn so line style + marker match exactly
    leg_ax = _legend_ax[0]
    if leg_ax is not None and leg_ax.get_legend() is not None:
        leg = leg_ax.get_legend()
        handles = leg.legend_handles
        labels  = [t.get_text() for t in leg.get_texts()]
        leg.remove()
    else:
        handles, labels = [], []

    fig.legend(
        handles, labels,
        loc='outside lower center',
        ncol=len(handles),
        frameon=True,
        fontsize=20.5,
        title=None,
        markerscale=2.2,
        handlelength=3.15,
        borderpad=0.945,
        labelspacing=0.63,
    )

    out_path = 'icl_ppl_unified_figure.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
