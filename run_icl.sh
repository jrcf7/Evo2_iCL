#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT="evo2_icl_ppl.py"
TRAIN_FASTA="data/tata_n5000_min20_max100_train.fna"
TEST_FASTA="data/tata_n5000_min20_max100_test.fna"
OUTPUT_DIR="results_ppl_tata_n5000_min20_max100"
SEED=42

# --- Models and N-shot values to sweep ---
MODELS=(
    "evo2_7b"
)

N_SHOTS=(256)

# --- Run all combinations ---
total=$(( ${#MODELS[@]} * ${#N_SHOTS[@]} ))
current=0

for MODEL in "${MODELS[@]}"; do
    for N in "${N_SHOTS[@]}"; do
        current=$(( current + 1 ))
        echo "========================================="
        echo "[$current/$total] Model: $MODEL  |  N-shots: $N"
        echo "========================================="

        python "$PYTHON_SCRIPT" \
            --model        "$MODEL" \
            --n-shots      "$N" \
            --train-fasta  "$TRAIN_FASTA" \
            --test-fasta   "$TEST_FASTA" \
            --output-dir   "$OUTPUT_DIR" \
            --seed         "$SEED"

        if [ $? -ne 0 ]; then
            echo "WARNING: run failed for $MODEL / N=$N — continuing..."
        else
            echo "Done: results/$MODEL_${N}shot"
        fi

        echo ""
    done
done

echo "========================================="
echo "All $total combinations completed."
echo "========================================="