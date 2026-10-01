# ---------------------------------------------------------------------------
# Perplexity-based in-context learning with MedGemma / Gemma 4
# for sequence classification
#
# Adapted from the Evo2 ICL pipeline (icl_ppl_evo2.py).
#
# Supports:
#   - MedGemma  (google/medgemma-*)  — Gemma-2 backbone, text-only CausalLM
#   - Gemma 4   (google/gemma-4-*)   — multimodal weights, text-only inference
#                                      via AutoTokenizer (avoids the torchvision
#                                      dependency that AutoProcessor pulls in);
#                                      use_cache must stay True
#
# Usage example:
#   python icl_ppl_medgemma.py \
#       --train-fasta data/amr_train.fna \
#       --test-fasta  data/amr_test.fna  \
#       --model       google/gemma-4-31B-it \
#       --n-shots     16
# ---------------------------------------------------------------------------

# --- Environment / cache settings -------------------------------------------
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

# ---------------------------------------------------------------------------
import argparse
import csv
import math
import random
from collections import defaultdict

import torch
from Bio import SeqIO
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Perplexity-based in-context learning with MedGemma/Gemma4 for sequence classification."
)
parser.add_argument("--train-fasta",      default="data/tata_n5000_min20_max100_train.fna")
parser.add_argument("--test-fasta",       default="data/tata_n5000_min20_max100_test.fna")
parser.add_argument("--n-shots",          default=128,  type=int)
parser.add_argument("--model",            default="google/gemma-4-31B-it")
parser.add_argument("--seed",             default=42, type=int)
parser.add_argument("--output-dir",       default="results_ppl_tata_n5000_min20_max100")
parser.add_argument("--max-model-tokens", default=None, type=int)
parser.add_argument("--buffer-len",       default=24,  type=int)
parser.add_argument("--label-len",        default=24,  type=int)
parser.add_argument("--dtype",            default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
parser.add_argument("--attn-implementation", default=None,
                    choices=["eager", "flash_attention_2", "sdpa"],
                    help="Attention backend. Defaults to 'sdpa' for Gemma 4, 'eager' for MedGemma.")
args = parser.parse_args()

random.seed(args.seed)

# ---------------------------------------------------------------------------
# Model-family detection
# ---------------------------------------------------------------------------

def is_gemma4(model_id: str) -> bool:
    """Return True for any google/gemma-4-* checkpoint."""
    return "gemma-4" in model_id.lower()

GEMMA4 = is_gemma4(args.model)
print(f"Model family: {'Gemma 4' if GEMMA4 else 'MedGemma / Gemma 2'}")

# Default attention backend: sdpa works best for Gemma 4; keep eager for MedGemma
attn_impl = args.attn_implementation or ("sdpa" if GEMMA4 else "eager")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

_model_tag = args.model.split("/")[-1]
output_dir = os.path.abspath(
    os.path.join(args.output_dir, f"{_model_tag}_{args.n_shots}shot")
)
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ["tata", "neutral"]

NUCLEOTIDES = ["A", "C", "G", "T"]

BUFFER_LEN = args.buffer_len
LABEL_LEN  = args.label_len

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}
TORCH_DTYPE = DTYPE_MAP[args.dtype]

# ---------------------------------------------------------------------------
# Context window lookup
# ---------------------------------------------------------------------------

CONTEXT_WINDOW_LOOKUP = {
    # MedGemma
    "google/medgemma-4b":         8_192,
    "google/medgemma-4b-it":      8_192,
    "google/medgemma-27b":        8_192,
    "google/medgemma-27b-it":     8_192,
    # Gemma 4
    "google/gemma-4-e2b":        128_000,
    "google/gemma-4-e2b-it":     128_000,
    "google/gemma-4-e4b":        128_000,
    "google/gemma-4-e4b-it":     128_000,
    "google/gemma-4-26b-a4b":    256_000,
    "google/gemma-4-26b-a4b-it": 256_000,
    "google/gemma-4-31b":        256_000,
    "google/gemma-4-31b-it":     256_000,
}

# ---------------------------------------------------------------------------
# Tokenizer / processor loading
# ---------------------------------------------------------------------------
#
# Gemma 4 is a multimodal model; its HF integration uses AutoProcessor rather
# than AutoTokenizer.  AutoProcessor wraps the text tokenizer and exposes the
# same .encode() / .decode() interface, so the rest of the script is unchanged.
# MedGemma (Gemma-2 backbone) ships as a plain text model and only needs
# AutoTokenizer.

print(f"\nLoading tokenizer: {args.model}...")

# AutoTokenizer works for both model families for text-only tokenisation.
# We deliberately avoid AutoProcessor here: Gemma 4's processor eagerly imports
# Gemma4VideoProcessor which requires torchvision — an unnecessary dependency
# for a pipeline that only feeds raw nucleotide strings.
# Note: mm_token_type_ids is validated by Gemma 4 only in model.train() mode,
# not in model.eval() / torch.no_grad() inference, so we never need to supply it.
tokenizer = AutoTokenizer.from_pretrained(
    args.model, use_fast=True
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

print(f"Loading model: {args.model}  (dtype={args.dtype}, attn={attn_impl})...")

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    dtype=TORCH_DTYPE,
    device_map="auto",
    attn_implementation=attn_impl,
)
model.eval()

# IMPORTANT for Gemma 4: setting use_cache=False corrupts logits due to a
# known bug in the sliding-window / global-attention hybrid.  We explicitly
# pass use_cache=True in every forward call.  Gemma4Config does not expose
# use_cache as a config attribute, so we check the generation_config instead
# and emit a warning rather than hard-crashing.
if GEMMA4:
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and getattr(gen_cfg, "use_cache", True) is False:
        print(
            "WARNING: generation_config.use_cache is False — this corrupts "
            "Gemma 4 logits. Forcing it back to True."
        )
        gen_cfg.use_cache = True

# ---------------------------------------------------------------------------
# Context window detection
# ---------------------------------------------------------------------------

def get_model_max_tokens(model_id: str) -> int:
    key = model_id.lower()
    if key in CONTEXT_WINDOW_LOOKUP:
        n = CONTEXT_WINDOW_LOOKUP[key]
        print(f"  Context window resolved from lookup table: {n} tokens")
        return n
    cfg_val = getattr(model.config, "max_position_embeddings", None)
    if cfg_val:
        print(f"  Context window resolved from model.config.max_position_embeddings: {cfg_val}")
        return cfg_val
    print("  WARNING: could not determine context window — falling back to 8192")
    return 8_192

MAX_MODEL_TOKENS = (
    args.max_model_tokens
    if args.max_model_tokens is not None
    else get_model_max_tokens(args.model)
)
print(f"Using context window: {MAX_MODEL_TOKENS} tokens")

# ---------------------------------------------------------------------------
# Input device detection
# ---------------------------------------------------------------------------
# With device_map="auto" the model's hf_device_map tells us exactly which
# device holds each named module.  The embedding layer is always on the first
# GPU — that is where we must place input_ids tensors.
#
# Strategy (in order):
#   1. Read hf_device_map and return the first non-cpu/non-disk device.
#   2. Fall back to the first CUDA parameter.
#   3. Last resort: first parameter regardless of device.

def get_input_device(mdl) -> torch.device:
    device_map = getattr(mdl, "hf_device_map", {})
    if device_map:
        for module_name, dev in device_map.items():
            if dev not in ("cpu", "disk"):
                resolved = torch.device(f"cuda:{dev}" if isinstance(dev, int) else dev)
                print(f"  Input device resolved from hf_device_map['{module_name}']: {resolved}")
                return resolved

    for name, param in mdl.named_parameters():
        if param.device.type == "cuda":
            print(f"  Input device resolved from first CUDA parameter '{name}': {param.device}")
            return param.device

    dev = next(mdl.parameters()).device
    print(f"  Input device resolved from first parameter: {dev}")
    return dev

# Diagnostic: dump the full hf_device_map so we can see what Accelerate assigned
_device_map = getattr(model, "hf_device_map", {})
print(f"  hf_device_map has {len(_device_map)} entries")
for _k, _v in list(_device_map.items())[:20]:
    print(f"    {_k!r}: {_v!r}")
if len(_device_map) > 20:
    print(f"    ... ({len(_device_map) - 20} more entries)")

input_device = get_input_device(model)
print(f"Using input device: {input_device}")

# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[int]:
    """
    Tokenize raw text to token IDs without special tokens.
    Works for both AutoTokenizer (MedGemma) and the .tokenizer attribute
    extracted from AutoProcessor (Gemma 4).
    """
    return tokenizer.encode(text, add_special_tokens=False)

# ---------------------------------------------------------------------------
# Nucleotide label scheme
# ---------------------------------------------------------------------------

def sample_label_nucleotides() -> tuple[str, str, str]:
    chosen = random.sample(NUCLEOTIDES, 3)
    return chosen[0], chosen[1], chosen[2]


def make_label_string(
    buffer_nuc: str,
    label_nuc: str,
    buffer_len: int | None = None,
    label_len: int | None  = None,
) -> str:
    if buffer_len is None:
        buffer_len = BUFFER_LEN
    if label_len is None:
        label_len = LABEL_LEN
    return buffer_nuc * buffer_len + label_nuc * label_len

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_few_shot_prefix(
    context_examples: list[tuple[str, int]],
    buffer_nuc: str,
    label_nuc_0: str,
    label_nuc_1: str,
    buffer_len: int | None = None,
    label_len: int | None  = None,
) -> str:
    if buffer_len is None:
        buffer_len = BUFFER_LEN
    if label_len is None:
        label_len = LABEL_LEN
    label_nucs = [label_nuc_0, label_nuc_1]
    parts = []
    for seq, class_idx in context_examples:
        suffix = make_label_string(buffer_nuc, label_nucs[class_idx], buffer_len, label_len)
        parts.append(seq + suffix)
    return "".join(parts)


def build_candidate_prompt(
    prefix: str,
    query_seq: str,
    buffer_nuc: str,
    candidate_label_nuc: str,
    buffer_len: int | None = None,
    label_len: int | None  = None,
) -> str:
    if buffer_len is None:
        buffer_len = BUFFER_LEN
    if label_len is None:
        label_len = LABEL_LEN
    suffix = make_label_string(buffer_nuc, candidate_label_nuc, buffer_len, label_len)
    return prefix + query_seq + suffix


def truncate_prefix_to_fit(
    context_examples: list[tuple[str, int]],
    query_seq: str,
    buffer_nuc: str,
    label_nuc_0: str,
    label_nuc_1: str,
    buffer_len: int | None = None,
    label_len: int | None  = None,
) -> tuple[list[tuple[str, int]], str]:
    if buffer_len is None:
        buffer_len = BUFFER_LEN
    if label_len is None:
        label_len = LABEL_LEN

    def prompt_len(seq_len: int) -> int:
        trunc_ctx   = [(s[:seq_len], ci) for s, ci in context_examples]
        trunc_query = query_seq[:seq_len]
        prefix      = build_few_shot_prefix(trunc_ctx, buffer_nuc, label_nuc_0, label_nuc_1, buffer_len, label_len)
        candidate   = build_candidate_prompt(prefix, trunc_query, buffer_nuc, label_nuc_0, buffer_len, label_len)
        return len(tokenize(candidate))

    full_prefix    = build_few_shot_prefix(context_examples, buffer_nuc, label_nuc_0, label_nuc_1, buffer_len, label_len)
    full_candidate = build_candidate_prompt(full_prefix, query_seq, buffer_nuc, label_nuc_0, buffer_len, label_len)
    if len(tokenize(full_candidate)) <= MAX_MODEL_TOKENS:
        return context_examples, query_seq

    print("  WARNING: prompt exceeds context window — truncating sequences...")

    max_seq_len = max(
        max((len(s) for s, _ in context_examples), default=32),
        len(query_seq),
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
            "Try reducing --n-shots."
        )

    trunc_ctx   = [(s[:final_len], ci) for s, ci in context_examples]
    trunc_query = query_seq[:final_len]
    return trunc_ctx, trunc_query

# ---------------------------------------------------------------------------
# Perplexity scoring
# ---------------------------------------------------------------------------

def compute_label_nll(
    input_ids_tensor: torch.Tensor,
    label_len: int | None = None,
) -> tuple[float, float]:
    """
    Run one forward pass and return (mean_NLL, perplexity) over the final
    `label_len` token positions.

    Gemma 4 note
    ------------
    use_cache=True is REQUIRED (use_cache=False corrupts logits in Gemma 4
    due to a bug in the sliding-window / global-attention hybrid).
    mm_token_type_ids is only validated by Gemma 4 during model.train(); it
    is NOT required for model.eval() / torch.no_grad() inference.
    """
    if label_len is None:
        label_len = LABEL_LEN

    input_ids_tensor = input_ids_tensor.long().to(input_device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids_tensor, use_cache=True)
        logits  = outputs.logits   # (1, seq_len, vocab_size)

    assert logits.ndim == 3, (
        f"Expected logits shape (1, seq_len, vocab_size), got {tuple(logits.shape)}."
    )

    log_probs = torch.log_softmax(logits[0], dim=-1)   # (seq_len, vocab_size)

    target_ids     = input_ids_tensor[0, -label_len:]
    pred_log_probs = log_probs[-label_len - 1 : -1]

    token_log_probs = pred_log_probs[
        torch.arange(label_len, device=pred_log_probs.device),
        target_ids,
    ]

    nll        = -token_log_probs.mean().item()
    perplexity = math.exp(nll)
    return nll, perplexity

# ---------------------------------------------------------------------------
# Load training sequences
# ---------------------------------------------------------------------------

print(f"\nLoading train sequences from {args.train_fasta}...")
train_pool: dict[str, list[tuple[str, str]]] = defaultdict(list)

for record in tqdm(SeqIO.parse(args.train_fasta, "fasta")):
    raw_label = next(
        (part.split("=")[1] for part in record.description.split() if part.startswith("label=")),
        None,
    )
    if raw_label not in CLASS_NAMES:
        continue
    train_pool[raw_label].append((record.id, str(record.seq).upper()))

for label, seqs in train_pool.items():
    print(f"  {label}: {len(seqs)} sequences available")

# ---------------------------------------------------------------------------
# Build all test items
# ---------------------------------------------------------------------------

print(f"\nPreparing test items...")
test_items = []

for record in tqdm(SeqIO.parse(args.test_fasta, "fasta")):
    raw_label = next(
        (part.split("=")[1] for part in record.description.split() if part.startswith("label=")),
        None,
    )
    if raw_label not in CLASS_NAMES:
        continue

    query_seq  = str(record.seq).upper()
    true_class = CLASS_NAMES.index(raw_label)

    context: list[tuple[str, int]] = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        pool    = train_pool[class_name]
        sampled = random.sample(pool, min(args.n_shots, len(pool)))
        for _, seq in sampled:
            context.append((seq, class_idx))
    random.shuffle(context)

    buffer_nuc, label_nuc_0, label_nuc_1 = sample_label_nucleotides()

    try:
        context, query_seq = truncate_prefix_to_fit(
            context, query_seq,
            buffer_nuc, label_nuc_0, label_nuc_1,
            BUFFER_LEN, LABEL_LEN,
        )
    except ValueError as e:
        print(f"  SKIPPING {record.id}: {e}")
        continue

    prefix   = build_few_shot_prefix(context, buffer_nuc, label_nuc_0, label_nuc_1, BUFFER_LEN, LABEL_LEN)
    prompt_0 = build_candidate_prompt(prefix, query_seq, buffer_nuc, label_nuc_0, BUFFER_LEN, LABEL_LEN)
    prompt_1 = build_candidate_prompt(prefix, query_seq, buffer_nuc, label_nuc_1, BUFFER_LEN, LABEL_LEN)

    ids_0 = tokenize(prompt_0)
    ids_1 = tokenize(prompt_1)

    test_items.append({
        "seq_id":      record.id,
        "raw_label":   raw_label,
        "true_class":  true_class,
        "ids_0":       ids_0,
        "ids_1":       ids_1,
        "buffer_nuc":  buffer_nuc,
        "label_nuc_0": label_nuc_0,
        "label_nuc_1": label_nuc_1,
    })

print(f"Total test items prepared: {len(test_items)}")

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

print(f"\nRunning perplexity-based inference...")

results = []
y_true: list[str] = []
y_pred: list[str] = []

for item in tqdm(test_items):
    seq_id     = item["seq_id"]
    true_class = item["true_class"]

    def score(ids: list[int]) -> tuple[float, float]:
        t = torch.tensor([ids], dtype=torch.long)
        return compute_label_nll(t, label_len=LABEL_LEN)

    nll_0, ppl_0 = score(item["ids_0"])
    nll_1, ppl_1 = score(item["ids_1"])

    pred_class = 0 if nll_0 <= nll_1 else 1
    pred_label = CLASS_NAMES[pred_class]
    true_label = CLASS_NAMES[true_class]

    y_true.append(true_label)
    y_pred.append(pred_label)

    results.append({
        "seq_id":      seq_id,
        "true_label":  true_label,
        "true_class":  true_class,
        "pred_label":  pred_label,
        "pred_class":  pred_class,
        "correct":     pred_class == true_class,
        "nll_class0":  nll_0,
        "nll_class1":  nll_1,
        "ppl_class0":  ppl_0,
        "ppl_class1":  ppl_1,
        "buffer_nuc":  item["buffer_nuc"],
        "label_nuc_0": item["label_nuc_0"],
        "label_nuc_1": item["label_nuc_1"],
    })

if not results:
    print("No results collected — check your FASTA labels and input files.")
    exit(1)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

observed_labels = [lbl for lbl in CLASS_NAMES if lbl in set(y_true) | set(y_pred)]
missing         = [lbl for lbl in CLASS_NAMES if lbl not in observed_labels]
if missing:
    print(f"  WARNING: classes never seen in predictions or ground truth: {missing}")

accuracy                       = sum(r["correct"] for r in results) / len(results)
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=observed_labels, zero_division=0
)
report = classification_report(y_true, y_pred, labels=observed_labels, zero_division=0)

print(f"\n{'='*50}")
print(f"Overall Accuracy: {accuracy:.4f}  ({sum(r['correct'] for r in results)}/{len(results)})")
print(f"\nPer-class metrics:")
print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 50)
for i, lbl in enumerate(observed_labels):
    print(f"{lbl:<12} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")
print(f"\nFull classification report:\n{report}")
print("=" * 50)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

predictions_path = os.path.join(output_dir, "predictions.tsv")
with open(predictions_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter="\t")
    writer.writeheader()
    writer.writerows(results)

metrics_path = os.path.join(output_dir, "metrics.tsv")
with open(metrics_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["class", "precision", "recall", "f1", "support"])
    for i, lbl in enumerate(observed_labels):
        writer.writerow([lbl, f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}", support[i]])
    writer.writerow(["accuracy", f"{accuracy:.4f}", "", "", len(results)])

report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"Overall Accuracy: {accuracy:.4f} ({sum(r['correct'] for r in results)}/{len(results)})\n\n")
    f.write(report)

print(f"\nResults saved to '{output_dir}/':")
print(f"  {predictions_path}")
print(f"  {metrics_path}")
print(f"  {report_path}")