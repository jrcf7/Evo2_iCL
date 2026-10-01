# --- Import and variable settings ---
import os

os.environ['CUDA_VISIBLE_DEVICES']  = '2,3,4,5,6,7'

import argparse
import random
import os
from collections import defaultdict
from Bio import SeqIO
from tqdm import tqdm
import csv
import math
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Perplexity-based in-context learning with Evo2 for sequence classification.")
parser.add_argument('--train-fasta',      default='data/amr_train.fna',   help='Path to train FASTA file (default: train.fna)')
parser.add_argument('--test-fasta',       default='data/amr_test.fna',    help='Path to test FASTA file (default: test.fna)')
parser.add_argument('--n-shots',          default=32,  type=int,     help='Number of examples per class to sample (default: 20)')
parser.add_argument('--model',            default='evo2_7b',          help='Evo2 model name (default: evo2_1b_base)')
parser.add_argument('--seed',             default=42,  type=int,     help='Random seed (default: 42)')
parser.add_argument('--output-dir',       default='results_ppl_AMR',     help='Output base directory (default: results_test)')
parser.add_argument('--max-model-tokens', default=None, type=int,    help='Override model context window size (default: auto-detect from model name)')
parser.add_argument('--buffer-len',       default=24,  type=int,     help='Length of the nucleotide buffer between sequence and label (default: 24)')
parser.add_argument('--label-len',        default=24,  type=int,     help='Length of the nucleotide label run (default: 24)')
args = parser.parse_args()

random.seed(args.seed)

# --- Output directory includes n-shots in name ---
output_dir = os.path.abspath(os.path.join(args.output_dir, f"{args.model}_{args.n_shots}shot"))
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# --- Class label names ---
# These are the human-readable class names expected in the FASTA label= field.
# Index 0 → class 0, index 1 → class 1.
CLASS_NAMES = ['tata', 'neutral']   # adapt as needed

# --- Evo2 model context window lookup ---
EVO2_CONTEXT_WINDOW = {
    'evo2_7b':       1_000_000,
    'evo2_40b':      1_000_000,
    'evo2_20b':      1_000_000,
    'evo2_7b_base':  8_192,
    'evo2_40b_base': 8_192,
    'evo2_1b_base':  8_192,
}

def get_model_max_tokens(model_name):
    """Return the context window size for a given Evo2 model name."""
    if model_name in EVO2_CONTEXT_WINDOW:
        tokens = EVO2_CONTEXT_WINDOW[model_name]
        print(f"  Context window resolved from lookup table: {tokens} tokens")
        return tokens
    print(f"  WARNING: Unknown model '{model_name}', falling back to 8192 tokens")
    return 8_192

MAX_MODEL_TOKENS = args.max_model_tokens if args.max_model_tokens is not None else get_model_max_tokens(args.model)
print(f"Using context window: {MAX_MODEL_TOKENS} tokens")

BUFFER_LEN = args.buffer_len   # number of buffer nucleotides between a sequence and its label
LABEL_LEN  = args.label_len    # number of label nucleotides


# ---------------------------------------------------------------------------
# Nucleotide label scheme
# ---------------------------------------------------------------------------

NUCLEOTIDES = ['A', 'C', 'G', 'T']

def sample_label_nucleotides():
    """
    Sample three distinct nucleotides without replacement from {A, C, G, T}:
      - buffer_nuc  : used for the 24-nt spacer between a sequence and its label
      - label_nuc_0 : label nucleotide representing class 0
      - label_nuc_1 : label nucleotide representing class 1

    Re-sampling per trial ensures no fixed nucleotide is systematically
    associated with a particular class, eliminating nucleotide-bias artifacts.

    Returns
    -------
    buffer_nuc  : str   single nucleotide character for the buffer
    label_nuc_0 : str   single nucleotide character for class 0
    label_nuc_1 : str   single nucleotide character for class 1
    """
    chosen = random.sample(NUCLEOTIDES, 3)
    return chosen[0], chosen[1], chosen[2]


def make_label_string(buffer_nuc, label_nuc, buffer_len=None, label_len=None):
    """
    Construct the label suffix appended to each in-context training example.

    Structure:  [buffer_nuc × buffer_len] + [label_nuc × label_len]

    The buffer provides a clear separator between the biological sequence and
    the classification label, reducing any ambiguity at the junction.

    Parameters
    ----------
    buffer_nuc  : str   nucleotide character repeated buffer_len times
    label_nuc   : str   nucleotide character repeated label_len times (the class token)
    buffer_len  : int   length of the buffer region (defaults to BUFFER_LEN if None)
    label_len   : int   length of the label region  (defaults to LABEL_LEN  if None)

    Returns
    -------
    str : the full label suffix (buffer + label run)
    """
    if buffer_len is None:
        buffer_len = BUFFER_LEN
    if label_len is None:
        label_len = LABEL_LEN
    return buffer_nuc * buffer_len + label_nuc * label_len


# ---------------------------------------------------------------------------
# Train / test data loading
# ---------------------------------------------------------------------------

print(f"\nLoading train sequences from {args.train_fasta}...")
train_pool = defaultdict(list)   # class_name -> list of (seq_id, sequence)

for record in tqdm(SeqIO.parse(args.train_fasta, "fasta")):
    raw_label = next(
        (part.split('=')[1] for part in record.description.split() if part.startswith('label=')),
        None
    )
    if raw_label not in CLASS_NAMES:
        continue
    train_pool[raw_label].append((record.id, str(record.seq).upper()))

for label, seqs in train_pool.items():
    print(f"  {label}: {len(seqs)} sequences available")


# ---------------------------------------------------------------------------
# Evo2 model loading
# ---------------------------------------------------------------------------

print(f"\nLoading Evo2 model: {args.model}...")
from evo2 import Evo2

model = Evo2(args.model)


def get_input_device(model):
    """
    Resolve the device (CPU/GPU) of the model's embedding layer so that
    input tensors can be placed on the correct device in multi-GPU setups.

    The outer Evo2 wrapper may not expose .device directly; we therefore
    walk through common embedding attribute names on the inner torch module.
    """
    inner = model.model
    for attr in ('embedding', 'embeddings', 'embed_tokens', 'tok_embeddings', 'encoder'):
        if hasattr(inner, attr):
            emb = getattr(inner, attr)
            if hasattr(emb, 'parameters'):
                device = next(emb.parameters()).device
                print(f"  Input device resolved from model.model.{attr}: {device}")
                return device
    device = next(inner.parameters()).device
    print(f"  Input device resolved from model.model first parameter: {device}")
    return device

device = get_input_device(model)
print(f"Using input device: {device}")


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def tokenize(text):
    """
    Tokenize a string using the Evo2 CharLevelTokenizer.

    Returns a list of integer token IDs (np.uint8 cast to int).
    Note: the tokenizer exposes `.tokenize(char)` rather than `.encode()`.
    """
    return [int(t) for t in model.tokenizer.tokenize(text)]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_few_shot_prefix(
    context_examples,
    buffer_nuc,
    label_nuc_0,
    label_nuc_1,
    buffer_len=None,
    label_len=None,
):
    """
    Construct the few-shot prefix from in-context training examples.

    Each example is: <sequence><buffer_nuc × buffer_len><label_nuc × label_len>

    The label nucleotide depends on the example's class:
      - class 0 → label_nuc_0
      - class 1 → label_nuc_1

    Parameters
    ----------
    context_examples : list of (sequence: str, class_idx: int)
        Balanced list of training sequences and their class indices (0 or 1).
    buffer_nuc       : str   buffer nucleotide (see sample_label_nucleotides)
    label_nuc_0      : str   label nucleotide for class 0
    label_nuc_1      : str   label nucleotide for class 1
    buffer_len  : int   length of buffer region (defaults to BUFFER_LEN if None)
    label_len   : int   length of label region  (defaults to LABEL_LEN  if None)

    Returns
    -------
    str : concatenated few-shot prefix (no separator between examples)
    """
    if buffer_len is None:
        buffer_len = BUFFER_LEN
    if label_len is None:
        label_len = LABEL_LEN
    label_nucs = [label_nuc_0, label_nuc_1]
    parts = []
    for seq, class_idx in context_examples:
        label_suffix = make_label_string(buffer_nuc, label_nucs[class_idx], buffer_len, label_len)
        parts.append(seq + label_suffix)
    return "".join(parts)


def build_candidate_prompt(prefix, query_seq, buffer_nuc, candidate_label_nuc,
                            buffer_len=None, label_len=None):
    """
    Build a complete candidate prompt for perplexity scoring.

    Structure:
        <few-shot prefix> + <query_seq> + <buffer> + <candidate label run>

    Two candidate prompts are built per test query: one for class 0 and one
    for class 1.  The model's perplexity is computed over the final label_len
    tokens (the label run only, not the buffer), and the candidate with lower
    perplexity is selected as the prediction.

    Parameters
    ----------
    prefix               : str   pre-built few-shot prefix
    query_seq            : str   the test sequence to classify
    buffer_nuc           : str   buffer nucleotide
    candidate_label_nuc  : str   label nucleotide for this candidate class
    buffer_len           : int   length of buffer region (defaults to BUFFER_LEN if None)
    label_len            : int   length of label region  (defaults to LABEL_LEN  if None)

    Returns
    -------
    str : full prompt string including the candidate label
    """
    if buffer_len is None:
        buffer_len = BUFFER_LEN
    if label_len is None:
        label_len = LABEL_LEN
    label_suffix = make_label_string(buffer_nuc, candidate_label_nuc, buffer_len, label_len)
    return prefix + query_seq + label_suffix


def truncate_prefix_to_fit(
    context_examples,
    query_seq,
    buffer_nuc,
    label_nuc_0,
    label_nuc_1,
    buffer_len=None,
    label_len=None,
):
    """
    Truncate all sequences so that the full candidate prompt fits within
    MAX_MODEL_TOKENS using binary search on per-sequence length.

    We test against the class-0 candidate prompt (both candidates have the
    same length, so checking one suffices).

    Parameters
    ----------
    context_examples : list of (sequence: str, class_idx: int)
    query_seq        : str
    buffer_nuc, label_nuc_0, label_nuc_1 : str   nucleotide characters
    buffer_len, label_len                : int   (default to BUFFER_LEN / LABEL_LEN if None)

    Returns
    -------
    (truncated_context_examples, truncated_query_seq) or raises ValueError
    """
    if buffer_len is None:
        buffer_len = BUFFER_LEN
    if label_len is None:
        label_len = LABEL_LEN
    def prompt_len(seq_len):
        trunc_ctx   = [(seq[:seq_len], ci) for seq, ci in context_examples]
        trunc_query = query_seq[:seq_len]
        prefix      = build_few_shot_prefix(trunc_ctx, buffer_nuc, label_nuc_0, label_nuc_1, buffer_len, label_len)
        candidate   = build_candidate_prompt(prefix, trunc_query, buffer_nuc, label_nuc_0, buffer_len, label_len)
        return len(tokenize(candidate))

    # Check if full-length prompt already fits
    full_prefix = build_few_shot_prefix(context_examples, buffer_nuc, label_nuc_0, label_nuc_1, buffer_len, label_len)
    full_prompt = build_candidate_prompt(full_prefix, query_seq, buffer_nuc, label_nuc_0, buffer_len, label_len)
    if len(tokenize(full_prompt)) <= MAX_MODEL_TOKENS:
        return context_examples, query_seq

    print(f"  WARNING: prompt exceeds context window — truncating sequences...")

    max_seq_len = max(
        max((len(seq) for seq, _ in context_examples), default=32),
        len(query_seq)
    )
    lo, hi = 32, max_seq_len
    while lo < hi:
        mid = (lo + hi) // 2
        if prompt_len(mid) <= MAX_MODEL_TOKENS:
            lo = mid + 1
        else:
            hi = mid

    final_len = lo - 1
    if final_len < 32:
        raise ValueError(
            f"Cannot fit prompt within {MAX_MODEL_TOKENS} tokens even at minimum sequence length. "
            f"Try reducing --n-shots."
        )

    trunc_ctx   = [(seq[:final_len], ci) for seq, ci in context_examples]
    trunc_query = query_seq[:final_len]
    return trunc_ctx, trunc_query


# ---------------------------------------------------------------------------
# Perplexity scoring
# ---------------------------------------------------------------------------

def compute_label_nll(input_ids_tensor, label_len=None):
    """
    Run a single forward pass and return the mean negative log-likelihood
    over the final `label_len` token positions.

    These positions correspond to the label nucleotide run (not the buffer),
    which is what we want to score: how well does the model predict the label
    nucleotides given the prompt?

    Lower NLL = lower perplexity = model more confident in this label.

    Parameters
    ----------
    input_ids_tensor : torch.Tensor   shape (1, seq_len), dtype torch.int32
    label_len        : int            number of label tokens to score
                                      (defaults to LABEL_LEN if None)

    Returns
    -------
    nll        : float   mean negative log-likelihood over the label run tokens
    perplexity : float   e^NLL — interpretable bits-per-token scale; used for
                         logging and replication; ranking is identical to NLL
    """
    if label_len is None:
        label_len = LABEL_LEN

    with torch.no_grad():
        outputs, _ = model(input_ids_tensor)   # model() returns (outputs_tuple, embeddings)
        logits     = outputs[0]                # outputs[0] is the logits tensor: (batch, seq_len, vocab_size)

    # Validate that logits have the expected 3-D shape so any Evo2 API change
    # is caught immediately rather than producing silently wrong scores.
    assert logits.ndim == 3, (
        f"Expected logits of shape (batch, seq_len, vocab_size), got shape {tuple(logits.shape)} "
        f"with type {type(logits)}. Check the Evo2 model output format."
    )

    # Work on the single batch item
    # logits[0, t, :] is the next-token distribution after position t, i.e.
    # it predicts token at position t+1.
    # The label run occupies positions [-label_len:] of the sequence.
    # To score those positions we need the logits at [-label_len-1 : -1].
    log_probs = torch.log_softmax(logits[0], dim=-1)   # (seq_len, vocab_size)

    # Actual target token IDs for the label run
    target_ids     = input_ids_tensor[0, -label_len:]       # (label_len,)
    pred_log_probs = log_probs[-label_len - 1:-1]            # (label_len, vocab_size)

    # Gather the log-prob the model assigned to each correct label token
    token_log_probs = pred_log_probs[
        torch.arange(label_len, device=pred_log_probs.device),
        target_ids.long()
    ]   # (label_len,)

    nll         = -token_log_probs.mean().item()
    perplexity  = math.exp(nll)   # PPL = e^NLL; equivalent ranking to NLL but on an interpretable scale
    return nll, perplexity


# ---------------------------------------------------------------------------
# Build all test items
# ---------------------------------------------------------------------------

print(f"\nPreparing test items...")
test_items = []   # list of dicts with all info needed for inference

for record in tqdm(SeqIO.parse(args.test_fasta, "fasta")):
    raw_label = next(
        (part.split('=')[1] for part in record.description.split() if part.startswith('label=')),
        None
    )
    if raw_label not in CLASS_NAMES:
        continue

    query_seq   = str(record.seq).upper()
    true_class  = CLASS_NAMES.index(raw_label)   # 0 or 1

    # Sample n_shots examples per class for a balanced context
    context = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        pool    = train_pool[class_name]
        sampled = random.sample(pool, min(args.n_shots, len(pool)))
        for _, seq in sampled:
            context.append((seq, class_idx))

    random.shuffle(context)

    # Sample fresh nucleotide assignments for this trial.
    # Randomising per query prevents any nucleotide bias accumulating across
    # the evaluation set.
    buffer_nuc, label_nuc_0, label_nuc_1 = sample_label_nucleotides()

    # Truncate if necessary to fit within the context window
    try:
        context, query_seq = truncate_prefix_to_fit(
            context, query_seq,
            buffer_nuc, label_nuc_0, label_nuc_1,
            BUFFER_LEN, LABEL_LEN,
        )
    except ValueError as e:
        print(f"  SKIPPING {record.id}: {e}")
        continue

    # Pre-build the shared few-shot prefix (same for both candidate prompts)
    prefix = build_few_shot_prefix(context, buffer_nuc, label_nuc_0, label_nuc_1, BUFFER_LEN, LABEL_LEN)

    # Build the two candidate prompts (one per class)
    prompt_0 = build_candidate_prompt(prefix, query_seq, buffer_nuc, label_nuc_0, BUFFER_LEN, LABEL_LEN)
    prompt_1 = build_candidate_prompt(prefix, query_seq, buffer_nuc, label_nuc_1, BUFFER_LEN, LABEL_LEN)

    ids_0 = tokenize(prompt_0)
    ids_1 = tokenize(prompt_1)

    test_items.append({
        'seq_id':      record.id,
        'raw_label':   raw_label,
        'true_class':  true_class,
        'ids_0':       ids_0,
        'ids_1':       ids_1,
        'buffer_nuc':  buffer_nuc,
        'label_nuc_0': label_nuc_0,
        'label_nuc_1': label_nuc_1,
    })

print(f"Total test items prepared: {len(test_items)}")


# ---------------------------------------------------------------------------
# Inference: perplexity-based label selection
# ---------------------------------------------------------------------------

print(f"\nRunning perplexity-based inference...")

results = []
y_true  = []
y_pred  = []

for item in tqdm(test_items):
    seq_id     = item['seq_id']
    true_class = item['true_class']
    true_label = CLASS_NAMES[true_class]

    # Score both candidate completions; lower NLL / PPL = model prefers this label
    def score(ids):
        t = torch.tensor([ids], dtype=torch.int32).to(device)
        return compute_label_nll(t, label_len=LABEL_LEN)

    nll_0, ppl_0 = score(item['ids_0'])
    nll_1, ppl_1 = score(item['ids_1'])

    # Prediction: class whose label run has lower perplexity (= lower NLL)
    pred_class = 0 if nll_0 <= nll_1 else 1
    pred_label = CLASS_NAMES[pred_class]

    y_true.append(true_label)
    y_pred.append(pred_label)

    results.append({
        'seq_id':      seq_id,
        'true_label':  true_label,
        'true_class':  true_class,
        'pred_label':  pred_label,
        'pred_class':  pred_class,
        'correct':     pred_class == true_class,
        'nll_class0':  nll_0,
        'nll_class1':  nll_1,
        'ppl_class0':  ppl_0,
        'ppl_class1':  ppl_1,
        'buffer_nuc':  item['buffer_nuc'],
        'label_nuc_0': item['label_nuc_0'],
        'label_nuc_1': item['label_nuc_1'],
    })

# --- Guard against empty results ---
if not results:
    print("No results collected — check your FASTA labels and input files.")
    exit(1)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# Use only labels that actually appear in predictions to avoid silent
# zero-inflation of precision/recall for classes absent from the test set.
observed_labels = [lbl for lbl in CLASS_NAMES if lbl in set(y_true) | set(y_pred)]
label_names     = observed_labels
missing         = [lbl for lbl in CLASS_NAMES if lbl not in observed_labels]
if missing:
    print(f"  WARNING: the following classes were never seen in predictions or ground truth: {missing}")
accuracy    = sum(r['correct'] for r in results) / len(results)

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=label_names, zero_division=0
)

report = classification_report(
    y_true, y_pred, labels=label_names, zero_division=0
)

print(f"\n{'='*50}")
print(f"Overall Accuracy: {accuracy:.4f}  ({sum(r['correct'] for r in results)}/{len(results)})")
print(f"\nPer-class metrics:")
print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 50)
for i, lbl in enumerate(label_names):
    print(f"{lbl:<12} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")
print(f"\nFull classification report:\n{report}")
print('='*50)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

predictions_path = os.path.join(output_dir, 'predictions.tsv')
with open(predictions_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter='\t')
    writer.writeheader()
    writer.writerows(results)

metrics_path = os.path.join(output_dir, 'metrics.tsv')
with open(metrics_path, 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['class', 'precision', 'recall', 'f1', 'support'])
    for i, lbl in enumerate(label_names):
        writer.writerow([lbl, f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}", support[i]])
    writer.writerow(['accuracy', f"{accuracy:.4f}", '', '', len(results)])

report_path = os.path.join(output_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(f"Overall Accuracy: {accuracy:.4f} ({sum(r['correct'] for r in results)}/{len(results)})\n\n")
    f.write(report)

print(f"\nResults saved to '{output_dir}/':")
print(f"  {predictions_path}")
print(f"  {metrics_path}")
print(f"  {report_path}")