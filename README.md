# Evo2_iCL

Code for the paper [Evo 2 as a classification machine: evidence from in-context learning and mechanistic interpretability](https://www.biorxiv.org/content/10.64898/2026.09.30.755330v1) experiments, using [Evo2](https://github.com/ArcInstitute/evo2), a long-range genomic foundation model, for DNA sequence classification. The pipeline classifies sequences (e.g. AMR genes vs. neutral, TATA-box-containing vs. neutral) without any fine-tuning by prompting the model with few-shot examples at inference time.

The perplexity-based ICL strategy is implemented and can be applied to any binary classification task over FASTA-formatted nucleotide sequences:

| Strategy | Script | Description |
|---|---|---|
| Perplexity-based | `evo2_icl_ppl.py` | Appends a nucleotide label run after each example; classifies by scoring two candidate completions with NLL |

LLM baselines (MedGemma / Gemma 4) are provided in `llm_icl_ppl.py` and `icl_ppl_llmunified.py` using the same perplexity-based protocol.

---

## Repository structure

```
evo2_icl_ppl.py           # Perplexity ICL with Evo2
llm_icl_ppl.py            # Perplexity ICL with MedGemma / Gemma 4
icl_ppl_llmunified.py     # Unified LLM perplexity ICL runner
mdgmm_icl_ppl.py          # MedGemma-specific perplexity ICL

run_icl.sh                # Shell script to sweep models and n-shot values

generate_dna.py           # Simulate FASTA sequences with/without TATA-box motifs
generate_random_cds.py    # Sample length-matched neutral CDS from a genome pool
shuffle_sequences.py      # Dinucleotide-shuffle sequences (negative control)
fna_files_merge.py        # Merge FASTA files
train_test_fna_merge.py   # Produce train/test splits from merged FASTA files

JCBScope_evo2_utils.py    # Activation / logit-lens utilities for Evo2
LogitLnes_evo2_utils.py   # Logit-lens analysis helpers
evo2_xai_utils.py         # XAI (explainability) utilities
JCScope_evo2_utils_distplot.py  # Distribution plots for JCBScope

icl_ppl_analysis.ipynb    # Perplexity ICL results analysis
icl_ppl_general_results.ipynb  # Cross-model results summary
logitlens_analysis.ipynb  # Logit-lens / internal representation analysis
sparse_autoencoder.ipynb  # Sparse autoencoder probing
```

---

## Requirements

- Python ≥ 3.10
- [Evo2](https://github.com/ArcInstitute/evo2) and its dependencies (CUDA required)
- `biopython`, `torch`, `scikit-learn`, `tqdm`, `numpy`
- For LLM baselines: `transformers`
- A HuggingFace token with access to the Evo2 model weights

---

## Data format

Input files must be FASTA files where the sequence header contains a `label=` field:

```
>seq_001 label=mrna
ATGCGT...
>seq_002 label=neutral
TTGCCA...
```

The `label=` value must match one of the class names defined in the script (`CLASS_NAMES` or `LABEL_MAP`).

### Generating synthetic data

```bash
# Simulate sequences with/without TATA-box motifs
python generate_dna.py -n 5000 --min 20 --max 100 --add-tata -o tata_pos.fasta
python generate_dna.py -n 5000 --min 20 --max 100 -o tata_neg.fasta

# Sample length-matched neutral CDS from a genome
python generate_random_cds.py --seed 42

# Shuffle sequences as dinucleotide-matched negatives
python shuffle_sequences.py
```

---

## Usage

### Perplexity ICL (`evo2_icl_ppl.py`)

Classifies sequences by scoring two candidate completions (one per class) using NLL over a nucleotide label run. Label nucleotides are randomly sampled per query to prevent nucleotide-bias artifacts.

```bash
# Minimal
python evo2_icl_ppl.py --train-fasta train.fna --test-fasta test.fna

# Full control
python evo2_icl_ppl.py \
  --train-fasta      train.fna \
  --test-fasta       test.fna \
  --n-shots          32 \
  --model            evo2_7b \
  --seed             42 \
  --output-dir       results_ppl \
  --buffer-len       24 \
  --label-len        24
```

### Sweeping models and n-shot values

Edit `run_icl.sh` to set your FASTA paths, models, and n-shot values, then run:

```bash
bash run_icl.sh
```

### LLM baselines

```bash
python llm_icl_ppl.py \
  --train-fasta data/amr_train.fna \
  --test-fasta  data/amr_test.fna  \
  --model       google/gemma-4-31B-it \
  --n-shots     16
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--train-fasta` | `amr_train.fna` | Path to training FASTA |
| `--test-fasta` | `amr_test.fna` | Path to test FASTA |
| `--n-shots` | 20 / 32 | Examples per class in the prompt |
| `--model` | `evo2_7b` | Evo2 model name |
| `--seed` | 42 | Random seed |
| `--output-dir` | `results_test` | Output directory |
| `--max-model-tokens` | auto | Override context window (tokens) |
| `--buffer-len` | 24 | Nucleotides between sequence and label run |
| `--label-len` | 24 | Length of the nucleotide label run |

**Supported Evo2 models and their context windows:**

| Model | Context window |
|---|---|
| `evo2_7b` | 1,000,000 tokens |
| `evo2_40b` | 1,000,000 tokens |
| `evo2_7b_base` | 8,192 tokens |
| `evo2_40b_base` | 8,192 tokens |
| `evo2_1b_base` | 8,192 tokens |

---

## Outputs

Each run writes three files to `<output-dir>/<model>_<n-shots>shot/`:

| File | Content |
|---|---|
| `predictions.tsv` | Per-sequence predictions, true labels, and class probabilities / NLL scores |
| `metrics.tsv` | Per-class precision, recall, F1, and support |
| `classification_report.txt` | Full sklearn classification report |

---

## Prompt design

Examples are concatenated without a separator; each is followed by a buffer run and a label nucleotide run:

```
<seq_0><buffer_nuc × 24><label_nuc_0 × 24><seq_1><buffer_nuc × 24><label_nuc_1 × 24>...<query><buffer_nuc × 24><candidate_label_nuc × 24>
```

Label nucleotides (`buffer_nuc`, `label_nuc_0`, `label_nuc_1`) are sampled fresh for each test query from `{A, C, G, T}` without replacement to eliminate systematic nucleotide bias.

When a prompt exceeds the model's context window, all sequences are truncated to the maximum length that fits, determined by binary search.
