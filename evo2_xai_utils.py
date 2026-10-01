import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def evo2_logit_lens(evo2_model, sequence, return_probs=True):
    """
    Logit lens across all blocks of Evo2.
    Applies model.norm + model.unembed at each block's output.
    """
    model = evo2_model.model
    tokenizer = evo2_model.tokenizer
    device = next(model.parameters()).device
    dtype = next(model.norm.parameters()).dtype

    captured = {}

    def make_hook(i):
        def hook(module, inp, out):
            # blocks return (hidden, residual) tuples
            h = out[0] if isinstance(out, tuple) else out
            captured[i] = h.detach()
        return hook

    hooks = [
        block.register_forward_hook(make_hook(i))
        for i, block in enumerate(model.blocks)
    ]

    input_ids = torch.tensor(
        tokenizer.tokenize(sequence), dtype=torch.int
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    # Probe the unembed Lambda once to understand its output shape/type
    # Lambda in Vortex is typically: lambda x: F.linear(x, embedding_layer.weight)
    # i.e. it ties weights with the input embedding and returns a plain tensor
    def apply_logit_lens(hidden):
        """Apply final norm + unembed to a captured hidden state."""
        with torch.no_grad():
            hidden = hidden.to(dtype)
            normed = model.norm(hidden)          # RMSNorm -> same shape
            out = model.unembed(normed)           # Lambda -> logits or tuple
            # Handle Lambda returning a tuple (some Vortex versions do this)
            logits = out[0] if isinstance(out, tuple) else out
            logits = logits.float()              # [batch, seq_len, vocab_size]
            return logits[0]                     # [seq_len, vocab_size]

    results = {}
    for i, hidden in captured.items():
        logits = apply_logit_lens(hidden)
        results[i] = F.softmax(logits, dim=-1) if return_probs else logits

    return input_ids, results


def decode_evo2_tokens(token_ids):
    """Convert Evo2 byte token IDs back to characters."""
    return [chr(t) if 32 <= t < 127 else f'<{t}>' for t in token_ids]


def print_logit_lens_table(sequence, results):
    """Print top-1 predicted token at each (layer, position)."""
    input_chars = list(sequence)
    header = f"{'Layer':>6} | " + " ".join(f"{c:>3}" for c in input_chars)
    print(header)
    print("-" * len(header))

    for layer_idx in sorted(results.keys()):
        probs = results[layer_idx]               # [seq_len, vocab_size]
        top_ids = probs.argmax(dim=-1).tolist()
        top_chars = decode_evo2_tokens(top_ids)
        row = f"{layer_idx:>6} | " + " ".join(f"{c:>3}" for c in top_chars)
        print(row)


def plot_logit_lens_evo2(sequence, results, target_char='G'):
    target_id = ord(target_char)
    num_layers = len(results)
    seq_len = len(sequence)

    prob_matrix = np.zeros((num_layers, seq_len))
    for layer_idx, probs in sorted(results.items()):
        prob_matrix[layer_idx] = probs[:, target_id].cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(10, seq_len * 0.5), 6))
    im = ax.imshow(prob_matrix, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax, label=f'P("{target_char}")')
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(list(sequence), fontfamily='monospace')
    ax.set_xlabel('Position')
    ax.set_ylabel('Layer')
    ax.set_title(f'Evo2 Logit Lens — P("{target_char}") across layers')
    plt.tight_layout()
    plt.show()


def plot_logit_lens_evo2_annotated(sequence, results, target_char=None, cmap='viridis', save_path=None):
    num_layers = len(results)
    seq_len = len(sequence)
    sorted_layers = sorted(results.keys())

    top_char_matrix = []
    color_matrix = np.zeros((num_layers, seq_len))

    for row_idx, layer_idx in enumerate(sorted_layers):
        probs = results[layer_idx]
        top_ids = probs.argmax(dim=-1).tolist()
        top_chars = decode_evo2_tokens(top_ids)
        top_char_matrix.append(top_chars)

        if target_char is not None:
            color_matrix[row_idx] = probs[:, ord(target_char)].cpu().numpy()
        else:
            color_matrix[row_idx] = probs.max(dim=-1).values.cpu().numpy()

    final_preds = top_char_matrix[-1]

    # Correct target at position i = sequence[i+1], last position has no target
    correct_targets = list(sequence[1:]) + ['?']

    fig, ax = plt.subplots(
        figsize=(max(10, seq_len * 0.9), max(6, num_layers * 0.4)),
        constrained_layout=True,
    )

    norm = mcolors.Normalize(vmin=color_matrix.min(), vmax=color_matrix.max())
    cmap_obj = plt.get_cmap(cmap)
    im = ax.imshow(color_matrix, aspect='auto', cmap=cmap_obj, norm=norm, origin='lower')

    # Annotate cells
    for row_idx in range(num_layers):
        for col_idx in range(seq_len):
            char = top_char_matrix[row_idx][col_idx]
            score = color_matrix[row_idx, col_idx]
            text_color = 'black' if score > 0.6 else 'white'
            ax.text(
                col_idx, row_idx, char,
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color=text_color,
                fontfamily='monospace',
            )

    # Top axis: correct next token per position
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(range(seq_len))
    ax_top.set_xticklabels(
        correct_targets,
        fontfamily='monospace', fontsize=11, fontweight='bold',
    )
    ax_top.set_xlabel('Target (next token)', fontsize=11, labelpad=8)

    # Bottom axis: input sequence
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(
        [f"{c}\n(pos {i})" for i, c in enumerate(sequence)],
        fontfamily='monospace', fontsize=9,
    )
    ax.set_xlabel('Input', fontsize=11, labelpad=8)

    # Y axis
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"Layer {i}" for i in sorted_layers], fontsize=9)
    ax.set_ylabel('Layer', fontsize=11)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=30, pad=0.04)
    cbar.set_label(
        f'P("{target_char}")' if target_char else 'P(top-1 token)',
        fontsize=10,
    )

    if target_char is not None:
        suptitle = f'Evo2 Logit Lens — color: P("{target_char}"), annotation: top-1 token'
    else:
        suptitle = 'Evo2 Logit Lens — color: P(top-1 token), annotation: top-1 token'
    fig.suptitle(suptitle, fontsize=13)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_logit_lens_evo2_token_averaged(sequence, results, target_char=None, cmap='viridis', save_path=None):
    """
    Logit lens heatmap where activations are averaged across all token positions.
    
    X-axis: vocabulary tokens (top-K most probable on average across final layer)
    Y-axis: layers
    Color:  average probability
    Annotation: probability value
    """
    num_layers = len(results)
    sorted_layers = sorted(results.keys())

    # Determine which vocab tokens to show:
    # Use the top-K tokens by mean probability at the final layer
    TOP_K = 20
    final_probs = results[sorted_layers[-1]]          # [seq_len, vocab_size]
    mean_final = final_probs.mean(dim=0)              # [vocab_size]
    topk_ids = mean_final.topk(TOP_K).indices.tolist()
    topk_chars = decode_evo2_tokens(topk_ids)

    # Build color matrix: [num_layers, TOP_K]
    color_matrix = np.zeros((num_layers, TOP_K))
    for row_idx, layer_idx in enumerate(sorted_layers):
        probs = results[layer_idx]                    # [seq_len, vocab_size]
        mean_probs = probs.mean(dim=0)                # [vocab_size] — averaged across tokens
        for col_idx, tok_id in enumerate(topk_ids):
            color_matrix[row_idx, col_idx] = mean_probs[tok_id].item()

    # --- Plot ---
    fig, ax = plt.subplots(
        figsize=(max(12, TOP_K * 0.9), max(6, num_layers * 0.4)),
        constrained_layout=True,
    )

    norm = mcolors.Normalize(vmin=color_matrix.min(), vmax=color_matrix.max())
    cmap_obj = plt.get_cmap(cmap)
    im = ax.imshow(color_matrix, aspect='auto', cmap=cmap_obj, norm=norm, origin='lower')

    # Annotate each cell with the probability value
    for row_idx in range(num_layers):
        for col_idx in range(TOP_K):
            score = color_matrix[row_idx, col_idx]
            text_color = 'black' if score > (color_matrix.max() * 0.6) else 'white'
            ax.text(
                col_idx, row_idx,
                f'{score:.3f}',
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color=text_color,
                fontfamily='monospace',
            )

    # X axis: vocab token characters
    ax.set_xticks(range(TOP_K))
    ax.set_xticklabels(topk_chars, fontfamily='monospace', fontsize=11, fontweight='bold')
    ax.set_xlabel('Token (top-K by mean probability at final layer)', fontsize=11, labelpad=8)

    # Y axis: layers
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"Layer {i}" for i in sorted_layers], fontsize=9)
    ax.set_ylabel('Layer', fontsize=11)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=30, pad=0.04)
    cbar.set_label('Mean P(token) across positions', fontsize=10)

    fig.suptitle(
        f'Evo2 Logit Lens — mean probability across tokens\nSequence: {sequence}',
        fontsize=13,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_logit_lens_evo2_averaged(sequence, results, cmap='viridis', save_path=None):
    """
    Logit lens heatmap averaged across all token positions.
    Single column per layer showing the mean top-1 probability.
    Cell is annotated with the numeric mean probability only — 
    no token annotation since the value is an average, not a single position.
    """
    num_layers = len(results)
    sorted_layers = sorted(results.keys())

    top_scores = []  # mean of top-1 probability across positions, per layer

    for layer_idx in sorted_layers:
        probs = results[layer_idx]                      # [seq_len, vocab_size]
        top1_probs = probs.max(dim=-1).values           # [seq_len] — top-1 prob at each position
        top_scores.append(top1_probs.mean().item())     # scalar — averaged across positions

    color_matrix = np.array(top_scores).reshape(-1, 1)  # [num_layers, 1]

    fig, ax = plt.subplots(
        figsize=(3, max(6, num_layers * 0.4)),
        constrained_layout=True,
    )

    norm = mcolors.Normalize(vmin=color_matrix.min(), vmax=color_matrix.max())
    im = ax.imshow(color_matrix, aspect='auto', cmap=plt.get_cmap(cmap), norm=norm, origin='lower')

    # Annotate with numeric probability only
    for row_idx, score in enumerate(top_scores):
        text_color = 'black' if score > (color_matrix.max() * 0.6) else 'white'
        ax.text(
            0, row_idx,
            f'{score:.3f}',
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color=text_color,
            fontfamily='monospace',
        )

    ax.set_xticks([0])
    ax.set_xticklabels(['avg'], fontsize=10)
    ax.set_xlabel('Mean across positions', fontsize=10, labelpad=8)

    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"Layer {i}" for i in sorted_layers], fontsize=9)
    ax.set_ylabel('Layer', fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=30, pad=0.04)
    cbar.set_label('Mean P(top-1 token) across positions', fontsize=10)

    fig.suptitle(
        f'Evo2 Logit Lens\nMean top-1 probability across all positions',
        fontsize=11,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_logit_lens_multi_model(sequence, model_ids, cmap='viridis', save_path=None, agg='mean'):
    """
    Logit lens aggregated across positions, compared across multiple models.
    Each column = one model, each row = one layer.
    Color + annotation = aggregated P(top-1 token) ± std across all positions.

    Args:
        sequence:   DNA string
        model_ids:  list of Evo2 model identifiers
        cmap:       matplotlib colormap
        save_path:  if set, saves figure at 300 dpi
        agg:        'mean' or 'median'
    """
    import torch
    from evo2 import Evo2

    assert agg in ('mean', 'median'), "agg must be 'mean' or 'median'"
    agg_fn = torch.mean if agg == 'mean' else torch.median

    all_scores = {}    # {model_id: [agg_score_per_layer]}
    all_stds   = {}    # {model_id: [std_per_layer]}
    all_layers = {}    # {model_id: num_layers}

    for mdl_id in model_ids:
        print(f"Loading {mdl_id}...")
        model = Evo2(mdl_id)
        _, results = evo2_logit_lens(model, sequence)

        sorted_layers = sorted(results.keys())
        scores, stds = [], []

        for layer_idx in sorted_layers:
            probs = results[layer_idx]              # [seq_len, vocab_size]
            top1_probs = probs.max(dim=-1).values   # [seq_len]

            if agg == 'median':
                scores.append(top1_probs.median().item())
            else:
                scores.append(top1_probs.mean().item())

            stds.append(top1_probs.std().item())

        all_scores[mdl_id] = scores
        all_stds[mdl_id]   = stds
        all_layers[mdl_id] = len(sorted_layers)

        del model
        torch.cuda.empty_cache()

    max_layers = max(all_layers.values())
    n_models   = len(model_ids)

    color_matrix = np.full((max_layers, n_models), np.nan)
    std_matrix   = np.full((max_layers, n_models), np.nan)

    for col_idx, mdl_id in enumerate(model_ids):
        n = all_layers[mdl_id]
        color_matrix[:n, col_idx] = all_scores[mdl_id]
        std_matrix[:n,   col_idx] = all_stds[mdl_id]

    # --- Plot ---
    fig, ax = plt.subplots(
        figsize=(max(4, n_models * 2.5), max(6, max_layers * 0.4)),
        constrained_layout=True,
    )

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='lightgrey')

    norm = mcolors.Normalize(
        vmin=np.nanmin(color_matrix),
        vmax=np.nanmax(color_matrix),
    )
    im = ax.imshow(color_matrix, aspect='auto', cmap=cmap_obj, norm=norm, origin='lower')

    # Annotate: aggregated value on top line, ±std below
    for row_idx in range(max_layers):
        for col_idx in range(n_models):
            score = color_matrix[row_idx, col_idx]
            std   = std_matrix[row_idx, col_idx]
            if np.isnan(score):
                continue
            text_color = 'black' if score > (np.nanmax(color_matrix) * 0.6) else 'white'
            ax.text(
                col_idx, row_idx,
                f'{score:.3f}\n±{std:.3f}',
                ha='center', va='center',
                fontsize=8, fontweight='bold',
                color=text_color,
                fontfamily='monospace',
                linespacing=1.4,
            )

    # X axis: model names
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_ids, fontsize=10, fontweight='bold', rotation=15, ha='right')
    ax.set_xlabel('Model', fontsize=11, labelpad=8)

    # Y axis: layers
    ax.set_yticks(range(max_layers))
    ax.set_yticklabels([f"Layer {i}" for i in range(max_layers)], fontsize=9)
    ax.set_ylabel('Layer', fontsize=11)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=30, pad=0.04)
    cbar.set_label(f'{agg.capitalize()} P(top-1 token) across positions', fontsize=10)

    fig.suptitle(
        f'Evo2 Logit Lens — model comparison\n'
        f'{agg.capitalize()} ± std of top-1 probability across positions\n',
        fontsize=11,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()