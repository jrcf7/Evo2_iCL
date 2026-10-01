#!/usr/bin/env python3
"""
generate_dna.py — Simulate DNA sequences with normally distributed lengths.

Usage
-----
    python generate_dna.py -n 1000 --min 50 --max 300 --add-tata -o seqs.fasta
    python generate_dna.py -n 500 --min 10 --max 100 --seed 42 --format tsv
"""

import argparse
import sys, re
import numpy as np


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def _sample_tata_box(rng: np.random.Generator) -> str:
    """Sample a TATAWAWR motif: T-A-T-A-W-A-W-R
    W = A or T (equal prob), R = A or G (equal prob)
    """
    w1 = rng.choice(["A", "T"])
    w2 = rng.choice(["A", "T"])
    r  = rng.choice(["A", "G"])
    return f"TATA{w1}A{w2}{r}"


_TATA_CORE_RE = re.compile(r"TATA[AT]")   # TATAW — minimal diagnostic prefix


def generate_dna_sequences(
    n: int,
    length_min: int,
    length_max: int,
    add_tata: bool = False,
    seed: int | None = None,
) -> list[str]:
    if length_min >= length_max:
        raise ValueError("--min must be strictly less than --max")

    TATA_LEN = 8  # length of TATAWAWR
    if add_tata and length_min < TATA_LEN:
        raise ValueError(
            f"--min ({length_min}) is shorter than the TATAWAWR motif ({TATA_LEN} bp). "
            "Increase --min to at least 8."
        )

    rng = np.random.default_rng(seed)
    mu    = (length_min + length_max) / 2
    sigma = (length_max - length_min) / 6
    lengths = np.clip(
        np.round(rng.normal(loc=mu, scale=sigma, size=n)),
        length_min,
        length_max,
    ).astype(int)

    bases = np.array(["A", "T", "G", "C"])
    sequences: list[str] = []

    for length in lengths:
        seq = list(rng.choice(bases, size=length))

        if add_tata:
            tata = _sample_tata_box(rng)
            pos  = int(rng.integers(0, length - TATA_LEN + 1))
            seq[pos : pos + TATA_LEN] = list(tata)
        else:
            # Scrub every TATAW (TATA[AT]) occurrence.
            # Strategy: resample the base at hit+4 (the W) to one of {C, G},
            # which is neither A nor T, destroying the motif without biasing flanks.
            non_w = ["C", "G"]  # bases that are not W (A/T)
            seq_str = "".join(seq)
            start = 0
            while True:
                m = _TATA_CORE_RE.search(seq_str, start)
                if m is None:
                    break
                hit = m.start()
                seq[hit + 4] = rng.choice(non_w)
                seq_str = "".join(seq)
                # Back up far enough to catch any overlap with the patched region
                start = max(0, hit - 4)

        sequences.append("".join(seq))

    return sequences

# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def write_fasta(sequences: list[str], out, prefix: str) -> None:
    for i, seq in enumerate(sequences, start=1):
        out.write(f">{prefix}_{i} len={len(seq)}\n")
        # Wrap at 60 characters (standard FASTA)
        for chunk_start in range(0, len(seq), 60):
            out.write(seq[chunk_start : chunk_start + 60] + "\n")


def write_tsv(sequences: list[str], out, prefix: str) -> None:
    out.write("id\tlength\tsequence\n")
    for i, seq in enumerate(sequences, start=1):
        out.write(f"{prefix}_{i}\t{len(seq)}\t{seq}\n")


def write_plain(sequences: list[str], out, **_) -> None:
    for seq in sequences:
        out.write(seq + "\n")


WRITERS = {
    "fasta": write_fasta,
    "tsv":   write_tsv,
    "plain": write_plain,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_dna.py",
        description="Generate simulated DNA sequences with normally distributed lengths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "-n", "--num-sequences",
        type=int,
        required=True,
        metavar="N",
        help="Number of sequences to generate.",
    )
    p.add_argument(
        "--min",
        type=int,
        required=True,
        dest="length_min",
        metavar="BP",
        help="Minimum sequence length (bp).",
    )
    p.add_argument(
        "--max",
        type=int,
        required=True,
        dest="length_max",
        metavar="BP",
        help="Maximum sequence length (bp).",
    )
    p.add_argument(
        "--add-tata",
        action="store_true",
        default=False,
        help="Embed a TATAAA motif at a random position in each sequence.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--format",
        choices=WRITERS.keys(),
        default="fasta",
        help="Output format.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="seq",
        metavar="STR",
        help="Sequence ID prefix used in FASTA/TSV output.",
    )
    p.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Output file path. Writes to stdout if omitted.",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        default=True,
        help="Print summary statistics to stderr after generation.",
    )

    return p


def print_stats(sequences: list[str]) -> None:
    lengths = np.array([len(s) for s in sequences])
    print(
        f"\n── stats ──────────────────────────\n"
        f"  sequences : {len(sequences)}\n"
        f"  min len   : {lengths.min()}\n"
        f"  max len   : {lengths.max()}\n"
        f"  mean len  : {lengths.mean():.1f}\n"
        f"  std dev   : {lengths.std():.1f}\n"
        f"───────────────────────────────────",
        file=sys.stderr,
    )


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    try:
        sequences = generate_dna_sequences(
            n=args.num_sequences,
            length_min=args.length_min,
            length_max=args.length_max,
            add_tata=args.add_tata,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))

    writer = WRITERS[args.format]

    extensions = {"fasta": "fasta", "tsv": "tsv", "plain": "txt"}
    tata_tag   = "tata-yes" if args.add_tata else "tata-no"
    auto_name  = (
        f"dna_n{args.num_sequences}"
        f"_min{args.length_min}"
        f"_max{args.length_max}"
        f"_{tata_tag}"
        f".{extensions[args.format]}"
    )

    import os
    output_path = args.output if args.output else os.path.join("data", auto_name)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as fh:
        writer(sequences, fh, args.prefix)
    print(f"Wrote {len(sequences)} sequences to {output_path}", file=sys.stderr)

    if args.stats:
        print_stats(sequences)


if __name__ == "__main__":
    main()