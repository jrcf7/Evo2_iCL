import argparse
import random
from Bio import SeqIO
from tqdm import tqdm

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Split and merge FASTA files with labels.")
parser.add_argument('--target',      default='data/dna_n2500_min20_max100_tata-yes.fasta', help='Path to Target FASTA file')
parser.add_argument('--TL',          default='tata', help='name of the target label')
parser.add_argument('--neutral',     default='data/dna_n2500_min20_max100_tata-no.fasta',  help='Path to neutral/CDS FASTA file')
parser.add_argument('--train-out',   default='data/tata_n5000_min20_max100_train.fna', help='Output train FASTA (default: train.fna)')
parser.add_argument('--test-out',    default='data/tata_n5000_min20_max100_test.fna', help='Output test FASTA (default: test.fna)')
parser.add_argument('--train-meta',  default='data/tata_n5000_min20_max100_train_metadata.tsv', help='Output train metadata')
parser.add_argument('--test-meta',   default='data/tata_n5000_min20_max100_test_metadata.tsv', help='Output test metadata')
parser.add_argument('--train-ratio', default=0.8, type=float, help='Train split ratio (default: 0.8)')
parser.add_argument('--seed',        default=42,  type=int, help='Random seed (default: 42)')
args = parser.parse_args()

def split_and_label(fasta_file, label, train_ratio=0.8, seed=42):
    """Read a FASTA file, shuffle, and split into train/test."""
    records = list(SeqIO.parse(fasta_file, "fasta"))
    random.seed(seed)
    random.shuffle(records)
    split_idx = int(len(records) * train_ratio)
    train_records = records[:split_idx]
    test_records  = records[split_idx:]
    print(f"{fasta_file} [{label}]: {len(train_records)} train / {len(test_records)} test")
    return train_records, test_records

# --- Split each file ---
trg_train,  trg_test  = split_and_label(args.target,    label=args.TL,   train_ratio=args.train_ratio, seed=args.seed)
ntr_train,  ntr_test  = split_and_label(args.neutral,   label='neutral', train_ratio=args.train_ratio, seed=args.seed)

# --- Merge and write ---
def write_split(records_with_labels, fasta_out, meta_out):
    with open(fasta_out, 'w') as fa, open(meta_out, 'w') as meta:
        meta.write("seq_id\tlabel\tseq_length\n")
        for record, label in tqdm(records_with_labels):
            record.description = f"label={label} {record.description}".strip()
            SeqIO.write(record, fa, "fasta")
            meta.write(f"{record.id}\t{label}\t{len(record.seq)}\n")

train_pairs = (
    [(r, args.TL)     for r in trg_train]  +
    [(r, 'neutral') for r in ntr_train]
)
test_pairs = (
    [(r, args.TL)     for r in trg_test]  +
    [(r, 'neutral') for r in ntr_test]
)

random.shuffle(train_pairs)
random.shuffle(test_pairs)

print(f"\nWriting {len(train_pairs)} train sequences to {args.train_out}...")
write_split(train_pairs, args.train_out, args.train_meta)

print(f"Writing {len(test_pairs)} test sequences to {args.test_out}...")
write_split(test_pairs, args.test_out, args.test_meta)

print("\nDone!")
print(f"  Train FASTA : {args.train_out}")
print(f"  Train meta  : {args.train_meta}")
print(f"  Test FASTA  : {args.test_out}")
print(f"  Test meta   : {args.test_meta}")

# --- Then loop over the merged file as before ---
all_embeddings = {}
sequence_info  = []

for fasta_file, split_name in [(args.train_out, 'train'), (args.test_out, 'test')]:
    print(f"\nProcessing sequences from {fasta_file}...")
    for record in tqdm(SeqIO.parse(fasta_file, "fasta")):
        seq_id     = record.id
        sequence   = str(record.seq).upper()
        seq_length = len(sequence)
        label = next(
            (part.split('=')[1] for part in record.description.split() if part.startswith('label=')),
            'unknown'
        )
        sequence_info.append({'id': seq_id, 'label': label, 'split': split_name, 'length': seq_length})