"""
jacobian_scope_distribution.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    results = gather_scope_data(backbone, input_ids, N, mode, device)
    fig     = plot_scope_distribution(results)
    fig     = plot_scope_distribution_annotated(results, TL, NL, B, l)
    fig     = plot_scope_ridge(results, TL, NL, B, l)
"""

import gc
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm

import JCBScope_evo2_utils as JCBScope


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data gathering
# ─────────────────────────────────────────────────────────────────────────────

def gather_scope_data(
    backbone,
    input_ids: torch.Tensor,
    N: int,
    mode: str = "Temperature",
    unnormalized_logits: bool = True,
    device: torch.device = None,
) -> dict:
    """
    Compute grad_magnitude = ‖∇_{residual_s} loss‖ for every source position s,
    repeated for each of the last N valid target tokens.

    A valid target satisfies loss_position <= seq_len - 2 (needs a next token).
    Always returns exactly min(N, seq_len - 1) curves.

    Multi-GPU: with model-parallel backbones (e.g. evo2_7b across 2 GPUs),
    input_ids lives on cuda:0 but the embedding layer (and thus the residual
    injection point) may be on cuda:1. residual_s is placed on the embedding
    layer's own device; presence stays on input_ids' device.

    Returns
    -------
    dict:
        seq_len, decoded_tokens, target_positions,
        grad_magnitudes (dict[int -> np.ndarray[seq_len]]), mode
    """
    if mode not in ("Temperature", "Semantic"):
        raise ValueError(f"mode must be 'Temperature' or 'Semantic', got {mode!r}")

    input_device = input_ids.device
    if device is None:
        device = input_device

    seq_len = input_ids.size(1)
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")

    # Last N positions satisfying loss_position <= seq_len - 2
    max_target       = seq_len - 2
    all_valid        = list(range(0, max_target + 1))   # [0 .. seq_len-2]
    target_positions = all_valid[-N:]                   # always exactly min(N, seq_len-1)

    decoded_tokens  = [chr(t.item()) for t in input_ids[0]]
    embedding_layer = JCBScope.get_embedding_layer(backbone)
    d_model         = embedding_layer.weight.shape[1]

    # residual must be on the same device as the embedding layer so the hook
    # injection is device-consistent, regardless of model sharding
    embedding_device = embedding_layer.weight.device

    # presence travels with input_ids (consumed before any cross-device transfer)
    presence = torch.ones(seq_len, dtype=torch.float32, device=input_device)

    hidden_norm_as_loss = (mode == "Temperature")
    grad_magnitudes     = {}

    for loss_pos in tqdm(target_positions, desc=f"{mode} sweep ({len(target_positions)} targets)"):
        grad_norms = []
        for s in range(seq_len):
            residual_s = nn.Parameter(torch.zeros(d_model, device=embedding_device))
            fp_s = JCBScope.customize_forward_pass(
                backbone=backbone, residual=residual_s,
                presence=presence, input_ids=input_ids, grad_idx=s,
            )
            loss_s, _ = fp_s(
                loss_position=loss_pos,
                hidden_norm_as_loss=hidden_norm_as_loss,
                unnormalized_logits=unnormalized_logits,
                tie_input_output_embed=False,
            )
            g = torch.autograd.grad(loss_s, residual_s)[0].detach().cpu()
            grad_norms.append(g.norm().item())

        grad_magnitudes[loss_pos] = np.array(grad_norms)
        torch.cuda.empty_cache()
        gc.collect()

    return dict(
        seq_len=seq_len,
        decoded_tokens=decoded_tokens,
        target_positions=target_positions,
        grad_magnitudes=grad_magnitudes,
        mode=mode,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Plain distribution plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_scope_distribution(
    results: dict,
    figsize: tuple = (10, 4),
    palette: str = "flare",
    fill_alpha: float = 0.25,
    line_alpha: float = 0.85,
    title: str = None,
) -> plt.Figure:
    """
    Filled line plot — one curve per target token.

    x-axis : source position index
    y-axis : ‖∇ residual‖ influence score
    hue    : one filled curve per target token
    """
    seq_len          = results["seq_len"]
    decoded_tokens   = results["decoded_tokens"]
    target_positions = results["target_positions"]
    grad_magnitudes  = results["grad_magnitudes"]
    mode             = results["mode"]

    x      = np.arange(seq_len)
    colors = sns.color_palette(palette, n_colors=len(target_positions))

    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    for color, loss_pos in zip(colors, target_positions):
        vec   = grad_magnitudes[loss_pos]
        label = f"pos {loss_pos} ({repr(decoded_tokens[loss_pos])})"
        ax.plot(x, vec, color=color, alpha=line_alpha, linewidth=1.5, label=label)
        ax.fill_between(x, vec, alpha=fill_alpha, color=color)

    ax.set_xlabel("Input Sequence Position")
    ax.set_ylabel("Influence")
    ax.set_title(title or f"{mode} Scope — influence on last {len(target_positions)} tokens",
                 fontweight="bold")
    ax.legend(title="Target token", loc="upper left", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Annotated distribution plot
# ─────────────────────────────────────────────────────────────────────────────

def _find_regions(decoded_tokens: list[str], nucleotide: str, l: int) -> list[tuple[int, int]]:
    """
    Find all runs of `nucleotide` that are >= l long.
    Returns list of (start, end) inclusive position pairs.
    """
    seq = "".join(decoded_tokens)
    runs, i, n = [], 0, len(seq)
    while i < n:
        if seq[i] == nucleotide:
            j = i
            while j < n and seq[j] == nucleotide:
                j += 1
            if j - i >= l:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def _find_tata_boxes(decoded_tokens: list[str]) -> list[tuple[int, int]]:
    """
    Find all occurrences of the TATA-box motif TATAWAWR
    where W = A or T, R = A or G.
    Returns list of (start, end) inclusive position pairs (length 8 each).
    """
    import re
    seq     = "".join(decoded_tokens)
    pattern = re.compile(r"(?=TATA[AT][AT][AG])")   # lookahead for overlapping matches
    return [(m.start(), m.start() + 7) for m in pattern.finditer(seq)]


def plot_scope_distribution_annotated(
    results: dict,
    TL: str,
    NL: str,
    B: str,
    l: int,
    figsize: tuple = (12, 4),
    palette: str = "flare",
    fill_alpha: float = 0.25,
    line_alpha: float = 0.85,
    buffer_color: str = "gold",
    neutral_label_color: str = "red",
    target_label_color: str = "green",
    region_alpha: float = 0.18,
    title: str = None,
) -> plt.Figure:
    """
    Filled-curve plot with shaded background regions marking buffer,
    target-label, and neutral-label nucleotide blocks.

    Parameters
    ----------
    results             : dict from gather_scope_data()
    TL                  : target label nucleotide character, e.g. 'C'
    NL                  : neutral label nucleotide character, e.g. 'T'
    B                   : buffer nucleotide character, e.g. 'A'
    l                   : length of each buffer / label block
    buffer_color        : shade colour for buffer regions
    neutral_label_color : shade colour for neutral label regions
    target_label_color  : shade colour for target label regions
    region_alpha        : transparency of the shaded bands
    """
    seq_len          = results["seq_len"]
    decoded_tokens   = results["decoded_tokens"]
    target_positions = results["target_positions"]
    grad_magnitudes  = results["grad_magnitudes"]
    mode             = results["mode"]

    x      = np.arange(seq_len)
    colors = sns.color_palette(palette, n_colors=len(target_positions))

    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    # ── Shaded regions (drawn first, behind curves) ───────────────────────────
    buffer_regions        = _find_regions(decoded_tokens, B,  l)
    target_label_regions  = _find_regions(decoded_tokens, TL, l)
    neutral_label_regions = _find_regions(decoded_tokens, NL, l)
    tata_regions          = _find_tata_boxes(decoded_tokens)

    for start, end in buffer_regions:
        ax.axvspan(start, end, color=buffer_color,        alpha=region_alpha, linewidth=0)
    for start, end in target_label_regions:
        ax.axvspan(start, end, color=target_label_color,  alpha=region_alpha, linewidth=0)
    for start, end in neutral_label_regions:
        ax.axvspan(start, end, color=neutral_label_color, alpha=region_alpha, linewidth=0)
    for start, end in tata_regions:
        ax.axvspan(start, end, color="lightblue",         alpha=region_alpha, linewidth=0)

    # ── Curves ───────────────────────────────────────────────────────────────
    for color, loss_pos in zip(colors, target_positions):
        vec   = grad_magnitudes[loss_pos]
        label = f"pos {loss_pos} ({repr(decoded_tokens[loss_pos])})"
        ax.plot(x, vec, color=color, alpha=line_alpha, linewidth=1.5, label=label)
        ax.fill_between(x, vec, alpha=fill_alpha, color=color)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = []
    if buffer_regions:
        legend_handles.append(
            mpatches.Patch(color=buffer_color, alpha=region_alpha + 0.2,
                           label=f"Buffer ({B}×{l})")
        )
    if target_label_regions:
        legend_handles.append(
            mpatches.Patch(color=target_label_color, alpha=region_alpha + 0.2,
                           label=f"Target label ({TL}×{l})")
        )
    if neutral_label_regions:
        legend_handles.append(
            mpatches.Patch(color=neutral_label_color, alpha=region_alpha + 0.2,
                           label=f"Neutral label ({NL}×{l})")
        )
    if tata_regions:
        legend_handles.append(
            mpatches.Patch(color="lightblue", alpha=region_alpha + 0.2,
                           label="TATA-box (TATAWAWR)")
        )

    ax.set_xlabel("Input Sequence Position")
    ax.set_ylabel("Influence")
    ax.set_title(
        title or f"{mode} Scope — annotated (B={B}, TL={TL}, NL={NL}, l={l})",
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ridge plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_scope_ridge(
    results: dict,
    TL: str,
    NL: str,
    B: str,
    l: int,
    row_height: float = 1.0,
    overlap: float = 0.6,
    figsize_width: float = 12,
    palette: str = "flare",
    fill_alpha: float = 0.8,
    line_alpha: float = 0.95,
    buffer_color: str = "gold",
    neutral_label_color: str = "red",
    target_label_color: str = "green",
    region_alpha: float = 0.25,
    title: str = None,
) -> plt.Figure:
    """
    Seaborn-style ridge (joy) plot — one shifted row per target token,
    with buffer, target-label, and neutral-label regions as shaded bands.

    Parameters
    ----------
    results             : dict from gather_scope_data()
    TL                  : target label nucleotide character, e.g. 'C'
    NL                  : neutral label nucleotide character, e.g. 'T'
    B                   : buffer nucleotide character, e.g. 'A'
    l                   : length of each buffer / label block
    row_height          : height in inches of each row axes
    overlap             : fraction of row_height that each row overlaps the one above
    figsize_width       : total figure width in inches
    palette             : colour palette (one colour per row)
    fill_alpha          : transparency of the filled area under each curve
    line_alpha          : transparency of the curve line
    buffer_color        : shade colour for buffer regions
    neutral_label_color : shade colour for neutral label regions
    target_label_color  : shade colour for target label regions
    region_alpha        : transparency of the region shading
    title               : overall figure title
    """
    seq_len          = results["seq_len"]
    decoded_tokens   = results["decoded_tokens"]
    target_positions = results["target_positions"]
    grad_magnitudes  = results["grad_magnitudes"]
    mode             = results["mode"]

    N      = len(target_positions)
    x      = np.arange(seq_len)
    colors = sns.color_palette(palette, n_colors=N)

    buffer_regions        = _find_regions(decoded_tokens, B,  l)
    target_label_regions  = _find_regions(decoded_tokens, TL, l)
    neutral_label_regions = _find_regions(decoded_tokens, NL, l)
    tata_regions          = _find_tata_boxes(decoded_tokens)

    fig_height = row_height * N * (1 - overlap) + row_height * overlap
    fig = plt.figure(figsize=(figsize_width, max(fig_height, 2)), dpi=120)

    gs   = gridspec.GridSpec(N, 1, hspace=-overlap, figure=fig)
    axes = [fig.add_subplot(gs[i]) for i in range(N)]

    ymax = max(grad_magnitudes[p].max() for p in target_positions) * 1.15

    for idx, (ax, loss_pos) in enumerate(zip(axes, target_positions)):
        vec   = grad_magnitudes[loss_pos]
        color = colors[idx]
        label = f"pos {loss_pos} ({repr(decoded_tokens[loss_pos])})"

        # Shaded regions behind curves
        for start, end in buffer_regions:
            ax.axvspan(start, end, color=buffer_color,        alpha=region_alpha, linewidth=0)
        for start, end in target_label_regions:
            ax.axvspan(start, end, color=target_label_color,  alpha=region_alpha, linewidth=0)
        for start, end in neutral_label_regions:
            ax.axvspan(start, end, color=neutral_label_color, alpha=region_alpha, linewidth=0)
        for start, end in tata_regions:
            ax.axvspan(start, end, color="lightblue",         alpha=region_alpha, linewidth=0)

        # White base fill so overlapping rows occlude rows below cleanly
        ax.fill_between(x, vec, color="white", alpha=1.0, zorder=1)

        # Filled curve + line
        ax.fill_between(x, vec, color=color, alpha=fill_alpha, zorder=2)
        ax.plot(x, vec, color=color, alpha=line_alpha, linewidth=1.2, zorder=3)

        # Row label on the right
        ax.text(1.002, 0.3, label, transform=ax.transAxes,
                va="center", ha="left", color=color)

        ax.set_xlim(0, seq_len - 1)
        ax.set_ylim(0, ymax)
        ax.set_yticks([])
        ax.patch.set_alpha(0)

        # Only the bottom row gets x-axis ticks and label
        if idx < N - 1:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
        else:
            ax.set_xlabel("Input Sequence Position")

        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        title or f"{mode} Scope — ridge plot (B={B}, TL={TL}, NL={NL}, l={l})",
        fontweight="bold", y=1.01,
    )

    # Legend on the top axes
    legend_handles = []
    if buffer_regions:
        legend_handles.append(
            mpatches.Patch(color=buffer_color, alpha=region_alpha + 0.2,
                           label=f"Buffer ({B}×{l})")
        )
    if target_label_regions:
        legend_handles.append(
            mpatches.Patch(color=target_label_color, alpha=region_alpha + 0.2,
                           label=f"Target label ({TL}×{l})")
        )
    if neutral_label_regions:
        legend_handles.append(
            mpatches.Patch(color=neutral_label_color, alpha=region_alpha + 0.2,
                           label=f"Neutral label ({NL}×{l})")
        )
    if tata_regions:
        legend_handles.append(
            mpatches.Patch(color="lightblue", alpha=region_alpha + 0.2,
                           label="TATA-box (TATAWAWR)")
        )
    if legend_handles:
        axes[0].legend(handles=legend_handles, loc="upper left", framealpha=0.8)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_scope_heatmap(
    results: dict,
    TL: str,
    NL: str,
    B: str,
    l: int,
    figsize: tuple = (14, 4),
    cmap: str = "flare",
    buffer_color: str = "gold",
    neutral_label_color: str = "red",
    target_label_color: str = "green",
    region_alpha: float = 0.25,
    title: str = None,
) -> plt.Figure:
    """
    Seaborn heatmap — rows are target tokens, columns are source positions.
    Cell colour encodes the ‖∇ residual‖ influence score.
    Buffer, target-label, and neutral-label regions are marked with
    coloured bands above the heatmap.

    Parameters
    ----------
    results             : dict from gather_scope_data()
    TL                  : target label nucleotide character, e.g. 'C'
    NL                  : neutral label nucleotide character, e.g. 'T'
    B                   : buffer nucleotide character, e.g. 'A'
    l                   : length of each buffer / label block
    cmap                : seaborn/matplotlib colormap for influence values
    buffer_color        : colour for buffer region band
    neutral_label_color : colour for neutral label region band
    target_label_color  : colour for target label region band
    region_alpha        : transparency of the region bands
    """
    seq_len          = results["seq_len"]
    decoded_tokens   = results["decoded_tokens"]
    target_positions = results["target_positions"]
    grad_magnitudes  = results["grad_magnitudes"]
    mode             = results["mode"]

    N = len(target_positions)

    # Build matrix: rows = targets, cols = source positions
    matrix = np.stack([grad_magnitudes[p] for p in target_positions], axis=0)  # [N, seq_len]

    row_labels = [f"pos {p} ({repr(decoded_tokens[p])})" for p in target_positions]

    buffer_regions        = _find_regions(decoded_tokens, B,  l)
    target_label_regions  = _find_regions(decoded_tokens, TL, l)
    neutral_label_regions = _find_regions(decoded_tokens, NL, l)
    tata_regions          = _find_tata_boxes(decoded_tokens)

    # Two-row layout: thin annotation band on top, heatmap below
    fig, (ax_ann, ax_hm) = plt.subplots(
        2, 1, figsize=figsize, dpi=120,
        gridspec_kw={"height_ratios": [0.08, 1], "hspace": 0.02},
    )

    # ── Heatmap (draw first so its xlim is set before we shade) ──────────────
    sns.heatmap(
        matrix,
        ax=ax_hm,
        cmap=cmap,
        yticklabels=row_labels,
        xticklabels=False,
        cbar_kws={"label": "Influence  ‖∇ residual‖", "shrink": 0.8},
        linewidths=0,
    )
    ax_hm.set_xlabel("Input Sequence Position")
    ax_hm.set_ylabel("Target Token")
    ax_hm.tick_params(axis="y", rotation=0)

    # ── Shade regions on the heatmap axes (cell coords: col i spans i..i+1) ──
    for start, end in buffer_regions:
        ax_hm.axvspan(start, end + 1, color=buffer_color,        alpha=region_alpha, linewidth=0, zorder=2)
    for start, end in target_label_regions:
        ax_hm.axvspan(start, end + 1, color=target_label_color,  alpha=region_alpha, linewidth=0, zorder=2)
    for start, end in neutral_label_regions:
        ax_hm.axvspan(start, end + 1, color=neutral_label_color, alpha=region_alpha, linewidth=0, zorder=2)
    for start, end in tata_regions:
        ax_hm.axvspan(start, end + 1, color="lightblue",         alpha=region_alpha, linewidth=0, zorder=2)

    # ── Annotation band: mirror the same shading above the heatmap ───────────
    ax_ann.set_xlim(ax_hm.get_xlim())
    ax_ann.set_ylim(0, 1)
    ax_ann.axis("off")

    for start, end in buffer_regions:
        ax_ann.axvspan(start, end + 1, color=buffer_color,        alpha=region_alpha + 0.15)
    for start, end in target_label_regions:
        ax_ann.axvspan(start, end + 1, color=target_label_color,  alpha=region_alpha + 0.15)
    for start, end in neutral_label_regions:
        ax_ann.axvspan(start, end + 1, color=neutral_label_color, alpha=region_alpha + 0.15)
    for start, end in tata_regions:
        ax_ann.axvspan(start, end + 1, color="lightblue",         alpha=region_alpha + 0.15)

    # ── Legend for region bands ───────────────────────────────────────────────
    legend_handles = []
    if buffer_regions:
        legend_handles.append(
            mpatches.Patch(color=buffer_color, alpha=region_alpha + 0.2,
                           label=f"Buffer ({B}×{l})")
        )
    if target_label_regions:
        legend_handles.append(
            mpatches.Patch(color=target_label_color, alpha=region_alpha + 0.2,
                           label=f"Target label ({TL}×{l})")
        )
    if neutral_label_regions:
        legend_handles.append(
            mpatches.Patch(color=neutral_label_color, alpha=region_alpha + 0.2,
                           label=f"Neutral label ({NL}×{l})")
        )
    if tata_regions:
        legend_handles.append(
            mpatches.Patch(color="lightblue", alpha=region_alpha + 0.2,
                           label="TATA-box (TATAWAWR)")
        )
    if legend_handles:
        ax_ann.legend(handles=legend_handles, loc="upper left",
                      framealpha=0.8, ncol=len(legend_handles),
                      bbox_to_anchor=(0, 1.0))

    fig.suptitle(
        title or f"{mode} Scope — heatmap (B={B}, TL={TL}, NL={NL}, l={l})",
        fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    return fig