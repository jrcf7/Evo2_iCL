import gc
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from evo2 import Evo2


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def decode_evo2_tokens(token_ids):
    """Convert Evo2 byte token IDs back to characters."""
    return [chr(t) if 32 <= t < 127 else f'<{t}>' for t in token_ids]


def _get_device(module):
    """Safely get the device of a module, including Lambda layers."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return torch.device(f'cuda:{torch.cuda.device_count() - 1}')


def find_norm_and_unembed(model):
    """Find final norm and unembed layer by inspecting named children."""
    children = dict(model.named_children())

    norm = None
    for candidate in ['norm', 'norm_f', 'final_norm', 'ln_f', 'layer_norm']:
        if candidate in children:
            norm = children[candidate]
            print(f"Found norm at: model.{candidate}")
            break

    unembed = None
    for candidate in ['unembed', 'lm_head', 'head', 'output_projection']:
        if candidate in children:
            unembed = children[candidate]
            print(f"Found unembed at: model.{candidate}")
            break

    if norm is None or unembed is None:
        print("All children:", list(children.keys()))
        raise AttributeError(
            "Could not auto-detect norm or unembed. See children printed above."
        )

    return norm, unembed


def _copy_norm_to_cpu(norm):
    """
    Copy any RMSNorm/LayerNorm to CPU regardless of internal parameter name.
    Handles: .weight, .scale, .gamma, .g
    """
    params = dict(norm.named_parameters())

    if len(params) == 0:
        raise RuntimeError(
            f"Norm layer {type(norm)} has no parameters. "
            f"Attributes: {[a for a in dir(norm) if not a.startswith('_')]}"
        )

    scale      = None
    scale_name = None
    for candidate in ['weight', 'scale', 'gamma', 'g']:
        if candidate in params:
            scale      = params[candidate].detach().cpu().float()
            scale_name = candidate
            break

    if scale is None:
        scale_name, scale_param = next(iter(params.items()))
        scale = scale_param.detach().cpu().float()
        print(f"Warning: using first parameter '{scale_name}' as norm scale")

    eps = 1e-6
    for candidate in ['eps', 'variance_epsilon', 'epsilon', 'norm_eps']:
        if hasattr(norm, candidate):
            eps = getattr(norm, candidate)
            break

    print(f"Copied norm: param='{scale_name}', shape={scale.shape}, eps={eps}")

    class _CPURMSNorm(torch.nn.Module):
        def __init__(self, scale, eps):
            super().__init__()
            self.scale = torch.nn.Parameter(scale)
            self.eps   = eps

        def forward(self, x):
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            return (x / rms) * self.scale

    return _CPURMSNorm(scale, eps)


def _get_unembed_weight_cpu(sh):
    """
    In Vortex, unembed is a Lambda wrapping VocabParallelEmbedding.unembed,
    which is the same object as embedding_layer.
    The weight is embedding_layer.weight.
    """
    # Get the bound method's self — that's the VocabParallelEmbedding instance
    _, unembed = find_norm_and_unembed(sh)
    
    func = unembed.func  # bound method: VocabParallelEmbedding.unembed

    # The embedding module is func.__self__
    if hasattr(func, '__self__'):
        embedding_module = func.__self__
        print(f"Unembed is bound to: {type(embedding_module).__name__}")
        for name, param in embedding_module.named_parameters():
            if param.ndim == 2:
                print(f"Found weight '{name}': {param.shape} on {param.device}")
                return param.detach().cpu().float()

    # Fallback: embedding_layer directly
    if hasattr(sh, 'embedding_layer'):
        for name, param in sh.embedding_layer.named_parameters():
            if param.ndim == 2:
                print(f"Fallback embedding_layer.{name}: {param.shape}")
                return param.detach().cpu().float()

    raise RuntimeError("Could not find unembed weight.")


# ──────────────────────────────────────────────
# Core logit lens
# ──────────────────────────────────────────────

def evo2_logit_lens(evo2_model, sequence, return_probs=True):
    """
    Multi-GPU safe logit lens across all blocks of Evo2.
    All probing (norm + unembed) done on CPU to avoid device mismatches.

    Returns:
        input_ids: [1, seq_len] tensor
        results:   dict {layer_idx -> tensor [seq_len, vocab_size]} on CPU
    """
    model     = evo2_model.model
    tokenizer = evo2_model.tokenizer

    norm, _   = find_norm_and_unembed(model)
    norm_cpu  = _copy_norm_to_cpu(norm)
    unembed_w = _get_unembed_weight_cpu(model)   # [vocab_size, d_model]

    def probe(hidden_cpu):
        """Apply norm + unembed entirely on CPU."""
        with torch.no_grad():
            normed = norm_cpu(hidden_cpu)           # [1, seq, d_model]
            logits = F.linear(normed, unembed_w)    # [1, seq, vocab]
            return logits[0]                        # [seq, vocab]

    # Register hooks — move to CPU immediately
    captured = {}

    def make_hook(i):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[i] = h.detach().cpu().float()
        return hook

    hooks = [
        block.register_forward_hook(make_hook(i))
        for i, block in enumerate(model.blocks)
    ]

    embed_device = _get_device(model.embedding_layer)
    input_ids = torch.tensor(
        tokenizer.tokenize(sequence), dtype=torch.int
    ).unsqueeze(0).to(embed_device)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    results = {}
    for i, hidden in captured.items():
        logits    = probe(hidden)
        results[i] = F.softmax(logits, dim=-1) if return_probs else logits

    return input_ids, results


# ──────────────────────────────────────────────
# Per-position heatmap
# ──────────────────────────────────────────────

def plot_logit_lens_evo2_annotated(sequence, results, target_char=None, cmap='viridis', save_path=None, tick_fontsize=14, label_fontsize=16):
    num_layers    = len(results)
    seq_len       = len(sequence)
    sorted_layers = sorted(results.keys())

    top_char_matrix = []
    color_matrix    = np.zeros((num_layers, seq_len))

    for row_idx, layer_idx in enumerate(sorted_layers):
        probs     = results[layer_idx]
        top_ids   = probs.argmax(dim=-1).tolist()
        top_chars = decode_evo2_tokens(top_ids)
        top_char_matrix.append(top_chars)

        if target_char is not None:
            color_matrix[row_idx] = probs[:, ord(target_char)].cpu().numpy()
        else:
            color_matrix[row_idx] = probs.max(dim=-1).values.cpu().numpy()

    correct_targets = list(sequence[1:]) + ['?']

    fig, ax = plt.subplots(
        figsize=(max(10, seq_len * 0.9), max(6, num_layers * 0.4)),
        constrained_layout=True,
    )

    norm_plot = mcolors.Normalize(vmin=color_matrix.min(), vmax=color_matrix.max())
    im = ax.imshow(color_matrix, aspect='auto', cmap=plt.get_cmap(cmap),
                   norm=norm_plot, origin='lower')

    for row_idx in range(num_layers):
        for col_idx in range(seq_len):
            char       = top_char_matrix[row_idx][col_idx]
            score      = color_matrix[row_idx, col_idx]
            text_color = 'black' if score > (color_matrix.max() * 0.6) else 'white'
            ax.text(col_idx, row_idx, char, ha='center', va='center',
                    fontsize=tick_fontsize, fontweight='bold', color=text_color,
                    fontfamily='monospace')

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(range(seq_len))
    ax_top.set_xticklabels(correct_targets, fontfamily='monospace',
                            fontsize=tick_fontsize, fontweight='bold')
    ax_top.set_xlabel('Target (next token)', fontsize=label_fontsize, labelpad=8)

    ax.set_xticks(range(seq_len))
    ax.set_xticklabels([f"{c}\n({i})" for i, c in enumerate(sequence)],
                       fontfamily='monospace', fontsize=tick_fontsize)
    ax.set_xlabel('Input', fontsize=label_fontsize, labelpad=8)

    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"Layer {i}" for i in sorted_layers], fontsize=tick_fontsize)
    ax.set_ylabel('Layer', fontsize=label_fontsize)

    cbar = fig.colorbar(im, ax=ax, shrink=1, aspect=25, pad=0.02)
    cbar.set_label(f'P("{target_char}")' if target_char else 'P(top-1 token)', fontsize=label_fontsize)

    fig.suptitle(
        f'Evo2 Logit Lens — {"P(" + target_char + ")" if target_char else "top-1 probability"}',
        fontsize=13,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


# ──────────────────────────────────────────────
# Averaged heatmap (single column)
# ──────────────────────────────────────────────

def plot_logit_lens_evo2_averaged(sequence, results, annotate=True, cmap='viridis', save_path=None, orientation='vertical', tick_fontsize=14, label_fontsize=16):
    """
    Single-column (vertical) or single-row (horizontal) heatmap.
    Mean P(top-1 token) across all positions per layer.
    Annotated with numeric probability only.

    Args:
        orientation: 'vertical'   — layers on Y axis, single column (default)
                     'horizontal' — layers on X axis, single row
    """
    assert orientation in ('vertical', 'horizontal'), \
        "orientation must be 'vertical' or 'horizontal'"

    sorted_layers = sorted(results.keys())
    num_layers    = len(sorted_layers)
    top_scores    = []

    for layer_idx in sorted_layers:
        probs      = results[layer_idx]
        top1_probs = probs.max(dim=-1).values
        top_scores.append(top1_probs.mean().item())

    scores_arr = np.array(top_scores)

    # Shape: [num_layers, 1] for vertical, [1, num_layers] for horizontal
    color_matrix = scores_arr.reshape(-1, 1) if orientation == 'vertical' \
                   else scores_arr.reshape(1, -1)

    # Figure size flips between orientations
    if orientation == 'vertical':
        figsize = (3, max(6, num_layers * 0.4))
    else:
        figsize = (max(6, num_layers * 0.4), 3)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    norm_plot = mcolors.Normalize(vmin=color_matrix.min(), vmax=color_matrix.max())
    im = ax.imshow(color_matrix, aspect='auto', cmap=plt.get_cmap(cmap),
                   norm=norm_plot, origin='lower')

    # Annotate cells
    if annotate:
        for idx, score in enumerate(top_scores):
            text_color = 'black' if score > (color_matrix.max() * 0.6) else 'white'
            col, row = (0, idx) if orientation == 'vertical' else (idx, 0)
            ax.text(col, row, f'{score:.3f}',
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color=text_color, fontfamily='monospace')

    if orientation == 'vertical':
        ax.set_xticks([0])
        ax.set_xticklabels(['avg'], fontsize=10)
        ax.set_xlabel('Mean across positions', fontsize=10, labelpad=8)
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([f"Layer {i}" for i in sorted_layers], fontsize=9)
        ax.set_ylabel('Layer', fontsize=11)
    else:
        ax.set_yticks([0])
        ax.set_yticklabels(['avg'], fontsize=10)
        ax.set_ylabel('Mean across positions', fontsize=10, labelpad=8)
        ax.set_xticks(range(num_layers))
        ax.set_xticklabels([f"L{i}" for i in sorted_layers],
                           fontsize=9, rotation=45, ha='right')
        ax.set_xlabel('Layer', fontsize=11, labelpad=8)

    cbar = fig.colorbar(im, ax=ax, shrink=1, aspect=25, pad=0.02)
    cbar.set_label('Mean P(top-1 token) across positions', fontsize=label_fontsize)

    fig.suptitle(
        f'Evo2 Logit Lens\nMean top-1 probability across all positions', fontsize=11,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

# ──────────────────────────────────────────────
# Multi-model comparison heatmap
# ──────────────────────────────────────────────

def plot_logit_lens_multi_model(sequence, model_ids, annotate=True, cmap='viridis', save_path=None, agg='mean', orientation='vertical', tick_fontsize=14, label_fontsize=18):
    """
    Multi-GPU safe model comparison logit lens.

    Args:
        agg:         'mean' or 'median'
        orientation: 'vertical'   — layers on Y axis, models on X axis (default)
                     'horizontal' — layers on X axis, models on Y axis
    """
    assert agg in ('mean', 'median'), "agg must be 'mean' or 'median'"
    assert orientation in ('vertical', 'horizontal'), \
        "orientation must be 'vertical' or 'horizontal'"

    # --- Model name conversion table ---
    model_name_map = {
        'evo2_1b_base':  'Evo2 1B Base',
        'evo2_7b':       'Evo2 7B',
        'evo2_40b':      'Evo2 40B',
        'evo2_7b_base':  'Evo2 7B Base',
        'evo2_40b_base': 'Evo2 40B Base',
    }
    display_ids = [model_name_map.get(m, m) for m in model_ids]

    all_scores = {}
    all_stds   = {}
    all_layers = {}

    for mdl_id in model_ids:
        print(f"\nLoading {mdl_id}...")
        evo2_model = Evo2(mdl_id)
        _, results = evo2_logit_lens(evo2_model, sequence)

        sorted_layers = sorted(results.keys())
        scores, stds  = [], []

        for layer_idx in sorted_layers:
            probs      = results[layer_idx]
            top1_probs = probs.max(dim=-1).values
            scores.append(
                top1_probs.median().item() if agg == 'median'
                else top1_probs.mean().item()
            )
            stds.append(top1_probs.std().item())

        all_scores[mdl_id] = scores
        all_stds[mdl_id]   = stds
        all_layers[mdl_id] = len(sorted_layers)

        del evo2_model, results
        gc.collect()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        print(f"  Freed {mdl_id} from all GPUs.")

    max_layers = max(all_layers.values())
    n_models   = len(model_ids)

    if orientation == 'vertical':
        color_matrix = np.full((max_layers, n_models), np.nan)
        std_matrix   = np.full((max_layers, n_models), np.nan)
        for col_idx, mdl_id in enumerate(model_ids):
            n = all_layers[mdl_id]
            color_matrix[:n, col_idx] = all_scores[mdl_id]
            std_matrix[:n,   col_idx] = all_stds[mdl_id]
    else:
        color_matrix = np.full((n_models, max_layers), np.nan)
        std_matrix   = np.full((n_models, max_layers), np.nan)
        for row_idx, mdl_id in enumerate(model_ids):
            n = all_layers[mdl_id]
            color_matrix[row_idx, :n] = all_scores[mdl_id]
            std_matrix[row_idx,   :n] = all_stds[mdl_id]

    if orientation == 'vertical':
        figsize = (max(4, n_models * 2.5), max(6, max_layers * 0.4))
    else:
        figsize = (max(6, max_layers * 0.4), max(4, n_models * 2.5))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='lightgrey')

    norm_plot = mcolors.Normalize(
        vmin=np.nanmin(color_matrix),
        vmax=np.nanmax(color_matrix),
    )
    im = ax.imshow(color_matrix, aspect='auto', cmap=cmap_obj,
                   norm=norm_plot, origin='lower')

    if annotate:
        nrows, ncols = color_matrix.shape
        for row_idx in range(nrows):
            for col_idx in range(ncols):
                score = color_matrix[row_idx, col_idx]
                std   = std_matrix[row_idx, col_idx]
                if np.isnan(score):
                    continue
                text_color = 'black' if score > (np.nanmax(color_matrix) * 0.6) else 'white'
                ax.text(col_idx, row_idx, f'{score:.3f}±{std:.3f}',
                        ha='center', va='center', fontsize=tick_fontsize, fontweight='bold',
                        color=text_color, fontfamily='monospace', linespacing=1.4,
                        rotation=90 if orientation == 'horizontal' else 0)

    if orientation == 'vertical':
        # X axis: models — Y axis: layers
        ax.set_xticks(range(n_models))
        ax.set_xticklabels(display_ids, fontsize=tick_fontsize, fontweight='bold')
        ax.set_xlabel('Model', fontsize=label_fontsize, labelpad=8)
        ax.set_yticks(range(max_layers))
        ax.set_yticklabels([f"Layer {i}" for i in range(max_layers)], fontsize=tick_fontsize)
        ax.set_ylabel('Layer', fontsize=label_fontsize)
    else:
        # X axis: layers — Y axis: models (vertical labels to save space)
        ax.set_xticks(range(max_layers))
        ax.set_xticklabels([f"L{i}" for i in range(max_layers)],
                           fontsize=tick_fontsize, rotation=45, ha='center')
        ax.set_xlabel('Layer', fontsize=label_fontsize, labelpad=8)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(display_ids, fontsize=tick_fontsize, fontweight='bold',
                           rotation=90, va='center')
        ax.set_ylabel('Model', fontsize=label_fontsize)

    cbar = fig.colorbar(im, ax=ax, shrink=1, aspect=25, pad=0.02)
    cbar.set_label(f'{agg.capitalize()} P(top-1 token) across positions', fontsize=label_fontsize)

    fig.suptitle(
        f'Evo2 Logit Lens — model comparison\n'
        f'{agg.capitalize()} ± std of top-1 probability across positions',
        fontsize=11,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
