# ---------------------------------------------------------------------------
# Unified Perplexity-based In-Context Learning for Sequence Classification
#
# Supports two inference backends, selected automatically from --model:
#
#   LOCAL (HuggingFace)   — any model loadable via AutoModelForCausalLM
#                           e.g. google/medgemma-4b, google/medgemma-27b-text-it
#
#   API (OpenAI-compatible) — any model served via an OpenAI-style REST API
#                             e.g. gpt-oos-120b, gpt-4o, meta-llama/...
#                             Requires --api-base and --api-key (or env vars).
#                             Uses token log-probabilities returned by the API
#                             (logprobs=True on the completions endpoint).
#
# Backend selection
# -----------------
# A model name is treated as API-based when it matches any prefix in
# API_MODEL_PREFIXES (see Constants section), or when --force-api is passed.
# Everything else is loaded locally via HuggingFace.
#
# Usage examples
# --------------
# Local (MedGemma):
#   python icl_ppl_unified.py \
#       --train-fasta data/amr_train.fna \
#       --test-fasta  data/amr_test.fna  \
#       --model       google/medgemma-4b  \
#       --n-shots     8
#
# API (GPT-OOS-120B or any OpenAI-compatible endpoint):
#   python icl_ppl_unified.py \
#       --train-fasta data/amr_train.fna \
#       --test-fasta  data/amr_test.fna  \
#       --model       gpt-oos-120b        \
#       --n-shots     8                   \
#       --api-base    https://your-endpoint/v1 \
#       --api-key     sk-...
#
# ---------------------------------------------------------------------------

# --- Environment / cache settings (local backend only) ----------------------
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# ---------------------------------------------------------------------------
import argparse
import csv
import math
import random
import time
from collections import defaultdict

from Bio import SeqIO
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ["tata", "neutral"]   # adapt as needed — must match label= in FASTA headers

NUCLEOTIDES = ["A", "C", "G", "T"]

# Model name prefixes (lowercased) that select the API backend automatically.
# Add any additional API-served model families here.
API_MODEL_PREFIXES = (
    "gpt-",
    "o1-",
    "o3-",
    "o4-",
    "text-davinci",
    "claude-",          # Anthropic via OpenAI-compatible proxy
    "mistral-",         # Mistral AI API
    "command-",         # Cohere API
    "gpt-oos-",         # custom OOS models
)

# Known HuggingFace context windows; auto-detected from config otherwise.
LOCAL_CONTEXT_WINDOW = {
    "google/medgemma-4b":            8_192,
    "google/medgemma-4b-it":         8_192,
    "google/medgemma-27b":           8_192,
    "google/medgemma-27b-it":        8_192,
    "google/medgemma-27b-text-it":   8_192,
}

# Default context window assumed for API models (tokens).
# The API backend never actually sends the full prompt to a tokenizer for
# length-checking; it relies on this value for truncation logic only.
API_DEFAULT_CONTEXT_WINDOW = 8_192

DTYPE_MAP = {
    "bfloat16": None,   # filled after torch import
    "float16":  None,
    "float32":  None,
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Unified perplexity-based ICL for sequence classification (local HF or API backend)."
)
# I/O
parser.add_argument("--train-fasta",      default="data/tata_n5000_min1000_max3000_train.fna")
parser.add_argument("--test-fasta",       default="data/tata_n5000_min1000_max3000_test.fna")
parser.add_argument("--output-dir",       default="results_ppl_unified")

# Model
parser.add_argument("--model",            default="google/medgemma-27b-text-it",
                    help="HuggingFace model ID (local) or API model name")
parser.add_argument("--force-api",        action="store_true",
                    help="Force API backend regardless of model name")
parser.add_argument("--force-local",      action="store_true",
                    help="Force local HF backend regardless of model name")

# API backend options
parser.add_argument("--api-base",         default=None,
                    help="Base URL for OpenAI-compatible API  (e.g. https://api.openai.com/v1). "
                         "Falls back to OPENAI_API_BASE env var.")
parser.add_argument("--api-key",          default=None,
                    help="API key. Falls back to OPENAI_API_KEY env var.")
parser.add_argument("--api-max-retries",  default=5,   type=int)
parser.add_argument("--api-retry-delay",  default=2.0, type=float,
                    help="Initial retry delay in seconds (doubles on each retry).")

# Local backend options
parser.add_argument("--dtype",            default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
parser.add_argument("--attn-implementation", default="eager",
                    choices=["eager", "flash_attention_2", "sdpa"])
parser.add_argument("--max-model-tokens", default=None, type=int,
                    help="Override context window size (default: auto-detect).")

# ICL options
parser.add_argument("--n-shots",          default=8,   type=int)
parser.add_argument("--buffer-len",       default=24,  type=int)
parser.add_argument("--label-len",        default=24,  type=int)
parser.add_argument("--seed",             default=42,  type=int)

args = parser.parse_args()

random.seed(args.seed)

BUFFER_LEN = args.buffer_len
LABEL_LEN  = args.label_len

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _is_api_model(model_id: str) -> bool:
    """Return True if the model should be served via the API backend."""
    if args.force_local:
        return False
    if args.force_api:
        return True
    return model_id.lower().startswith(API_MODEL_PREFIXES)

USE_API = _is_api_model(args.model)
print(f"\nModel  : {args.model}")
print(f"Backend: {'API (OpenAI-compatible)' if USE_API else 'Local (HuggingFace)'}")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

_model_tag = args.model.split("/")[-1]
output_dir = os.path.abspath(
    os.path.join(args.output_dir, f"{_model_tag}_{args.n_shots}shot")
)
os.makedirs(output_dir, exist_ok=True)
print(f"Output : {output_dir}")

# ---------------------------------------------------------------------------
# Backend initialisation
# ---------------------------------------------------------------------------

# ---- Local backend ---------------------------------------------------------

if not USE_API:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    DTYPE_MAP["bfloat16"] = torch.bfloat16
    DTYPE_MAP["float16"]  = torch.float16
    DTYPE_MAP["float32"]  = torch.float32
    TORCH_DTYPE = DTYPE_MAP[args.dtype]

    print(f"\nLoading tokenizer : {args.model}...")
    _tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    print(f"Loading model     : {args.model}  "
          f"(dtype={args.dtype}, attn={args.attn_implementation})...")
    _model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    _model.eval()

    # Context window
    def _get_local_max_tokens() -> int:
        if args.model in LOCAL_CONTEXT_WINDOW:
            n = LOCAL_CONTEXT_WINDOW[args.model]
            print(f"  Context window (lookup table): {n}")
            return n
        cfg_val = getattr(_model.config, "max_position_embeddings", None)
        if cfg_val:
            print(f"  Context window (model.config): {cfg_val}")
            return cfg_val
        print("  WARNING: context window unknown — defaulting to 8192")
        return 8_192

    MAX_MODEL_TOKENS = (
        args.max_model_tokens if args.max_model_tokens is not None
        else _get_local_max_tokens()
    )

    # Input device (first embedding layer)
    def _get_input_device() -> "torch.device":
        for attr in ("embed_tokens", "embedding", "embeddings", "tok_embeddings"):
            inner = getattr(_model, "model", _model)
            if hasattr(inner, attr):
                dev = next(getattr(inner, attr).parameters()).device
                print(f"  Input device ({attr}): {dev}")
                return dev
        dev = next(_model.parameters()).device
        print(f"  Input device (first param): {dev}")
        return dev

    _input_device = _get_input_device()
    print(f"Context window: {MAX_MODEL_TOKENS} tokens")

# ---- API backend -----------------------------------------------------------

else:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for the API backend.\n"
            "Install it with:  pip install openai"
        )

    _api_base = args.api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    _api_key  = args.api_key  or os.environ.get("OPENAI_API_KEY",  "")

    if not _api_key:
        raise ValueError(
            "No API key found. Pass --api-key or set the OPENAI_API_KEY environment variable."
        )

    _api_client = OpenAI(api_key=_api_key, base_url=_api_base)
    MAX_MODEL_TOKENS = args.max_model_tokens or API_DEFAULT_CONTEXT_WINDOW
    print(f"API base       : {_api_base}")
    print(f"Context window : {MAX_MODEL_TOKENS} tokens (override with --max-model-tokens)")

# ---------------------------------------------------------------------------
# Tokenizer helpers  (unified interface)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[int]:
    """
    Convert text to a list of token IDs.

    Local backend : uses the HF tokenizer (no special tokens).
    API backend   : approximates token count via the tiktoken cl100k_base
                    encoding (GPT-3.5 / GPT-4 family).  For non-OpenAI models
                    this is only used for context-window truncation; the actual
                    token IDs are never sent to the model.
    """
    if not USE_API:
        return _tokenizer.encode(text, add_special_tokens=False)
    else:
        # Lazy-import tiktoken; fall back to char/4 heuristic if unavailable.
        try:
            import tiktoken
            _enc = tiktoken.get_encoding("cl100k_base")
            return _enc.encode(text)
        except ImportError:
            # Rough approximation: 1 token ≈ 4 characters for English/nucleotide text
            return list(range(len(text) // 4))

# ---------------------------------------------------------------------------
# Nucleotide label scheme  (identical to original script)
# ---------------------------------------------------------------------------

def sample_label_nucleotides() -> tuple[str, str, str]:
    chosen = random.sample(NUCLEOTIDES, 3)
    return chosen[0], chosen[1], chosen[2]


def make_label_string(
    buffer_nuc: str,
    label_nuc: str,
    buffer_len: int | None = None,
    label_len:  int | None = None,
) -> str:
    if buffer_len is None: buffer_len = BUFFER_LEN
    if label_len  is None: label_len  = LABEL_LEN
    return buffer_nuc * buffer_len + label_nuc * label_len


# ---------------------------------------------------------------------------
# Prompt construction  (identical to original script)
# ---------------------------------------------------------------------------

def build_few_shot_prefix(
    context_examples: list[tuple[str, int]],
    buffer_nuc: str,
    label_nuc_0: str,
    label_nuc_1: str,
    buffer_len: int | None = None,
    label_len:  int | None = None,
) -> str:
    if buffer_len is None: buffer_len = BUFFER_LEN
    if label_len  is None: label_len  = LABEL_LEN
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
    label_len:  int | None = None,
) -> str:
    if buffer_len is None: buffer_len = BUFFER_LEN
    if label_len  is None: label_len  = LABEL_LEN
    suffix = make_label_string(buffer_nuc, candidate_label_nuc, buffer_len, label_len)
    return prefix + query_seq + suffix


def truncate_prefix_to_fit(
    context_examples: list[tuple[str, int]],
    query_seq: str,
    buffer_nuc: str,
    label_nuc_0: str,
    label_nuc_1: str,
    buffer_len: int | None = None,
    label_len:  int | None = None,
) -> tuple[list[tuple[str, int]], str]:
    if buffer_len is None: buffer_len = BUFFER_LEN
    if label_len  is None: label_len  = LABEL_LEN

    def prompt_len(seq_len: int) -> int:
        trunc_ctx   = [(s[:seq_len], ci) for s, ci in context_examples]
        trunc_query = query_seq[:seq_len]
        prefix    = build_few_shot_prefix(trunc_ctx, buffer_nuc, label_nuc_0, label_nuc_1, buffer_len, label_len)
        candidate = build_candidate_prompt(prefix, trunc_query, buffer_nuc, label_nuc_0, buffer_len, label_len)
        return len(tokenize(candidate))

    # Fast path
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

    return [(s[:final_len], ci) for s, ci in context_examples], query_seq[:final_len]


# ---------------------------------------------------------------------------
# Perplexity scoring — LOCAL backend
# ---------------------------------------------------------------------------

def _compute_label_nll_local(
    input_ids: list[int],
    label_len: int | None = None,
) -> tuple[float, float]:
    """
    One HF forward pass → mean NLL and perplexity over the final label_len tokens.
    """
    import torch

    if label_len is None:
        label_len = LABEL_LEN

    t = torch.tensor([input_ids], dtype=torch.long).to(_input_device)

    with torch.no_grad():
        logits = _model(input_ids=t).logits   # (1, seq_len, vocab_size)

    assert logits.ndim == 3
    log_probs = torch.log_softmax(logits[0], dim=-1)   # (seq_len, vocab_size)

    target_ids     = t[0, -label_len:]
    pred_log_probs = log_probs[-label_len - 1 : -1]

    token_log_probs = pred_log_probs[
        torch.arange(label_len, device=pred_log_probs.device),
        target_ids,
    ]
    nll = -token_log_probs.mean().item()
    return nll, math.exp(nll)


# ---------------------------------------------------------------------------
# Perplexity scoring — API backend
# ---------------------------------------------------------------------------

def _compute_label_nll_api(
    prompt_text: str,
    label_len: int | None = None,
) -> tuple[float, float]:
    """
    Call the completions endpoint with logprobs=True and echo=True, then
    extract the per-token log-probabilities for the trailing label run.

    The OpenAI legacy completions API (/v1/completions) supports `echo=True`
    which returns logprobs for the *prompt* tokens — this is what we need to
    score the label suffix in context.  The chat completions endpoint does NOT
    support echo; we use the legacy endpoint here.

    If the endpoint does not support echo (some third-party providers), we
    fall back to a zero-token completion and read the `prompt_logprobs` field
    if available (vLLM / Together style).
    """
    if label_len is None:
        label_len = LABEL_LEN

    for attempt in range(args.api_max_retries):
        try:
            response = _api_client.completions.create(
                model=args.model,
                prompt=prompt_text,
                max_tokens=0,          # we only want prompt logprobs
                echo=True,             # return logprobs for prompt tokens
                logprobs=0,            # request top-0 (just the chosen token)
                temperature=0,
            )
            break
        except Exception as exc:
            if attempt == args.api_max_retries - 1:
                raise
            wait = args.api_retry_delay * (2 ** attempt)
            print(f"  API error (attempt {attempt+1}): {exc}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    choice = response.choices[0]

    # ---- Extract token log-probabilities ------------------------------------
    # Standard OpenAI format: choice.logprobs.token_logprobs is a list of
    # floats (None for first token), one per prompt token.
    lp = choice.logprobs
    if lp is None or lp.token_logprobs is None:
        raise RuntimeError(
            "API response did not include logprobs. "
            "Ensure the endpoint supports echo=True and logprobs=0. "
            "For vLLM, pass --api-extra-body '{\"prompt_logprobs\": 1}' or use a compatible model."
        )

    token_logprobs: list[float | None] = lp.token_logprobs

    # The label run occupies the last `label_len` tokens of the prompt.
    # We take the last label_len entries; skip any None (first-token sentinel).
    label_logprobs = [lp for lp in token_logprobs[-label_len:] if lp is not None]

    if not label_logprobs:
        raise RuntimeError(
            f"Could not extract {label_len} label logprobs from API response. "
            f"Got {len(token_logprobs)} total token logprobs."
        )

    nll = -sum(label_logprobs) / len(label_logprobs)
    return nll, math.exp(nll)


# ---------------------------------------------------------------------------
# Unified scoring dispatcher
# ---------------------------------------------------------------------------

def compute_label_nll(
    prompt_text_or_ids,
    label_len: int | None = None,
) -> tuple[float, float]:
    """
    Dispatch to the correct backend.

    Parameters
    ----------
    prompt_text_or_ids : str  (API backend)  or  list[int]  (local backend)
    label_len          : number of label tokens to score
    """
    if USE_API:
        if not isinstance(prompt_text_or_ids, str):
            raise TypeError("API backend expects a prompt string, not token IDs.")
        return _compute_label_nll_api(prompt_text_or_ids, label_len)
    else:
        if not isinstance(prompt_text_or_ids, list):
            raise TypeError("Local backend expects a list of token IDs, not a string.")
        return _compute_label_nll_local(prompt_text_or_ids, label_len)


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
# Build test items
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

    # Balanced context: n_shots per class
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

    item: dict = {
        "seq_id":      record.id,
        "raw_label":   raw_label,
        "true_class":  true_class,
        "buffer_nuc":  buffer_nuc,
        "label_nuc_0": label_nuc_0,
        "label_nuc_1": label_nuc_1,
    }

    if USE_API:
        # Store raw strings — the API receives the full text
        item["prompt_0"] = prompt_0
        item["prompt_1"] = prompt_1
    else:
        # Pre-tokenise for the local backend
        item["ids_0"] = tokenize(prompt_0)
        item["ids_1"] = tokenize(prompt_1)

    test_items.append(item)

print(f"Total test items: {len(test_items)}")

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

print(f"\nRunning perplexity-based inference ({('API' if USE_API else 'local')})...")

results = []
y_true: list[str] = []
y_pred: list[str] = []

for item in tqdm(test_items):
    true_class = item["true_class"]

    if USE_API:
        nll_0, ppl_0 = compute_label_nll(item["prompt_0"], LABEL_LEN)
        nll_1, ppl_1 = compute_label_nll(item["prompt_1"], LABEL_LEN)
    else:
        nll_0, ppl_0 = compute_label_nll(item["ids_0"], LABEL_LEN)
        nll_1, ppl_1 = compute_label_nll(item["ids_1"], LABEL_LEN)

    pred_class = 0 if nll_0 <= nll_1 else 1
    pred_label = CLASS_NAMES[pred_class]
    true_label = CLASS_NAMES[true_class]

    y_true.append(true_label)
    y_pred.append(pred_label)

    results.append({
        "seq_id":      item["seq_id"],
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
        "backend":     "api" if USE_API else "local",
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
    print(f"  WARNING: classes never seen: {missing}")

accuracy                       = sum(r["correct"] for r in results) / len(results)
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=observed_labels, zero_division=0
)
report = classification_report(y_true, y_pred, labels=observed_labels, zero_division=0)

print(f"\n{'='*55}")
print(f"Backend : {'API' if USE_API else 'Local HF'}  |  Model: {args.model}")
print(f"Accuracy: {accuracy:.4f}  ({sum(r['correct'] for r in results)}/{len(results)})")
print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 55)
for i, lbl in enumerate(observed_labels):
    print(f"{lbl:<12} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")
print(f"\nFull classification report:\n{report}")
print("=" * 55)

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
    f.write(f"Model   : {args.model}\n")
    f.write(f"Backend : {'API' if USE_API else 'Local HF'}\n")
    f.write(f"Accuracy: {accuracy:.4f} ({sum(r['correct'] for r in results)}/{len(results)})\n\n")
    f.write(report)

print(f"\nResults saved to '{output_dir}/':")
print(f"  {predictions_path}")
print(f"  {metrics_path}")
print(f"  {report_path}")