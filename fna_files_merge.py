#!/usr/bin/env python3
"""
merge_fna.py — Merge two .fna (FASTA nucleotide) files into one,
preserving the original FASTA structure of both inputs.

Usage:
    python merge_fna.py file1.fna file2.fna output.fna [options]

Options:
    --shuffle           Randomly interleave records from both files
    --line-width N      Wrap sequence lines at N characters (default: preserve original)
    --deduplicate       Skip records whose sequence ID already appeared
    --prefix1 STR       Prefix to prepend to sequence IDs from file1
    --prefix2 STR       Prefix to prepend to sequence IDs from file2
    -v, --verbose       Print summary statistics after merging
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# FASTA I/O helpers
# ---------------------------------------------------------------------------

def parse_fasta(path: Path) -> Iterator[tuple[str, str, str]]:
    """
    Yield (header, seq_id, sequence) tuples for every record in a FASTA file.
    - header  : the full header line including '>'
    - seq_id  : the first token after '>' (up to the first whitespace)
    - sequence: the full, un-wrapped nucleotide sequence (uppercase)
    Handles both Unix and Windows line endings, and gzip-compressed files.
    """
    import gzip

    open_fn = gzip.open if str(path).endswith(".gz") else open

    header = None
    seq_chunks: list[str] = []

    with open_fn(path, "rt", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, header[1:].split()[0], "".join(seq_chunks).upper()
                header = line
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, header[1:].split()[0], "".join(seq_chunks).upper()


def write_record(fh, header: str, sequence: str, line_width: int | None) -> None:
    """Write a single FASTA record to an open file handle."""
    fh.write(header + "\n")
    if line_width is None or line_width <= 0:
        fh.write(sequence + "\n")
    else:
        for i in range(0, len(sequence), line_width):
            fh.write(sequence[i : i + line_width] + "\n")


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------

def merge_fna(
    path1: Path,
    path2: Path,
    out_path: Path,
    shuffle: bool = False,
    line_width: int | None = None,
    deduplicate: bool = False,
    prefix1: str = "",
    prefix2: str = "",
    verbose: bool = False,
) -> None:
    # Load all records from both files
    def load(path: Path, prefix: str) -> list[tuple[str, str, str]]:
        records = []
        for header, seq_id, seq in parse_fasta(path):
            if prefix:
                # Insert prefix right after '>'
                header = ">" + prefix + header[1:]
            records.append((header, seq_id, seq))
        return records

    records1 = load(path1, prefix1)
    records2 = load(path2, prefix2)

    all_records: list[tuple[str, str, str]]
    if shuffle:
        all_records = records1 + records2
        random.shuffle(all_records)
    else:
        all_records = records1 + records2

    seen_ids: set[str] = set()
    written = skipped = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for header, seq_id, seq in all_records:
            if deduplicate:
                if seq_id in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(seq_id)
            write_record(out, header, seq, line_width)
            written += 1

    if verbose:
        print(f"[merge_fna] Input file 1 : {path1}  ({len(records1):,} records)")
        print(f"[merge_fna] Input file 2 : {path2}  ({len(records2):,} records)")
        print(f"[merge_fna] Records written : {written:,}")
        if deduplicate:
            print(f"[merge_fna] Records skipped (duplicate IDs): {skipped:,}")
        print(f"[merge_fna] Output written to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two .fna (FASTA nucleotide) files into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("file1", type=Path, help="First .fna input file")
    parser.add_argument("file2", type=Path, help="Second .fna input file")
    parser.add_argument("output", type=Path, help="Merged output .fna file")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly interleave records from both files",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=None,
        metavar="N",
        help="Wrap sequence lines at N characters (default: preserve original width)",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Skip records whose sequence ID was already seen",
    )
    parser.add_argument(
        "--prefix1",
        default="",
        metavar="STR",
        help="Prefix to prepend to IDs from file1 (e.g. 'sample1_')",
    )
    parser.add_argument(
        "--prefix2",
        default="",
        metavar="STR",
        help="Prefix to prepend to IDs from file2 (e.g. 'sample2_')",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print summary statistics"
    )

    args = parser.parse_args()

    for p in (args.file1, args.file2):
        if not p.exists():
            print(f"Error: file not found — {p}", file=sys.stderr)
            sys.exit(1)

    merge_fna(
        path1=args.file1,
        path2=args.file2,
        out_path=args.output,
        shuffle=args.shuffle,
        line_width=args.line_width,
        deduplicate=args.deduplicate,
        prefix1=args.prefix1,
        prefix2=args.prefix2,
        verbose=args.verbose,
    )

    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()