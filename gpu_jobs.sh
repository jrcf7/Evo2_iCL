#!/bin/bash

# GPU Job Lister with Age Information
# Lists running processes on GPU(s) with their age/duration and program name

echo "=========================================="
echo "       GPU Jobs & Process Age Info        "
echo "=========================================="
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is an NVIDIA GPU present?"
    exit 1
fi

# Get GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
echo "GPUs detected: $GPU_COUNT"
echo ""

# Show GPU utilization summary
echo "--- GPU Utilization Summary ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits | \
while IFS=',' read -r idx name util mem_used mem_total temp; do
    printf "GPU %s | %-30s | Util: %3s%% | Mem: %s/%s MiB | Temp: %s°C\n" \
        "$idx" "$(echo $name | xargs)" "$(echo $util | xargs)" \
        "$(echo $mem_used | xargs)" "$(echo $mem_total | xargs)" "$(echo $temp | xargs)"
done

echo ""
echo "--- Processes Running on GPU ---"
echo ""

# Header
printf "%-8s %-8s %-10s %-20s %-22s %-15s %-8s\n" \
    "GPU" "PID" "MEM(MiB)" "AGE" "PROGRAM" "USER" "CPU%"
printf '%0.s-' {1..97}
echo ""

# Get GPU processes
GPU_PROCS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader,nounits 2>/dev/null)

if [ -z "$GPU_PROCS" ]; then
    echo "No compute processes currently running on any GPU."
else
    # Map GPU UUID to index
    declare -A UUID_TO_IDX
    while IFS=',' read -r idx uuid; do
        UUID_TO_IDX["$(echo $uuid | xargs)"]="$(echo $idx | xargs)"
    done < <(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader)

    while IFS=',' read -r gpu_uuid pid mem_used; do
        pid=$(echo "$pid" | xargs)
        gpu_uuid=$(echo "$gpu_uuid" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        gpu_idx="${UUID_TO_IDX[$gpu_uuid]:-?}"

        if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
            continue
        fi

        # --- Age calculation ---
        START_TIME=$(ps -o lstart= -p "$pid" 2>/dev/null | xargs)
        if [ -n "$START_TIME" ]; then
            START_EPOCH=$(date -d "$START_TIME" +%s 2>/dev/null)
            NOW_EPOCH=$(date +%s)
            AGE_SECS=$(( NOW_EPOCH - START_EPOCH ))
            DAYS=$(( AGE_SECS / 86400 ))
            HOURS=$(( (AGE_SECS % 86400) / 3600 ))
            MINS=$(( (AGE_SECS % 3600) / 60 ))
            SECS=$(( AGE_SECS % 60 ))
            if [ "$DAYS" -gt 0 ]; then
                AGE_STR="${DAYS}d ${HOURS}h ${MINS}m"
            elif [ "$HOURS" -gt 0 ]; then
                AGE_STR="${HOURS}h ${MINS}m ${SECS}s"
            elif [ "$MINS" -gt 0 ]; then
                AGE_STR="${MINS}m ${SECS}s"
            else
                AGE_STR="${SECS}s"
            fi
        else
            AGE_STR="unknown"
        fi

        # --- Process details ---
        USER=$(ps -o user= -p "$pid" 2>/dev/null | xargs)
        CPU=$(ps -o %cpu= -p "$pid" 2>/dev/null | xargs)

        # Full command line
        FULL_CMD=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' | xargs)
        if [ -z "$FULL_CMD" ]; then
            FULL_CMD=$(ps -o args= -p "$pid" 2>/dev/null | xargs)
        fi

        # --- Program name detection ---
        EXE_PATH=$(readlink -f /proc/$pid/exe 2>/dev/null)
        PROG_NAME=$(basename "$EXE_PATH" 2>/dev/null)
        [ -z "$PROG_NAME" ] && PROG_NAME=$(ps -o comm= -p "$pid" 2>/dev/null | xargs)

        CMD_LOWER=$(echo "$FULL_CMD" | tr '[:upper:]' '[:lower:]')

        # For Python: extract the .py script name (fixed regex - no variable-length lookbehind)
        if echo "$PROG_NAME" | grep -qi "python"; then
            SCRIPT=$(echo "$FULL_CMD" | grep -oE '[^ ]+\.py' | head -1)
            if [ -n "$SCRIPT" ]; then
                PROG_NAME="$(basename "$SCRIPT") [py]"
            else
                PROG_NAME="python"
            fi
        fi

        # Detect ML/GPU framework from command line
        FRAMEWORK=""
        case "$CMD_LOWER" in
            *torch*)      FRAMEWORK="PyTorch" ;;
            *tensorflow*) FRAMEWORK="TensorFlow" ;;
            *jax*)        FRAMEWORK="JAX" ;;
            *cuda*)       FRAMEWORK="CUDA" ;;
            *ollama*)     FRAMEWORK="Ollama" ;;
            *stable*)     FRAMEWORK="StableDiff" ;;
        esac
        [ -n "$FRAMEWORK" ] && PROG_DISPLAY="$PROG_NAME ($FRAMEWORK)" || PROG_DISPLAY="$PROG_NAME"

        # Truncate to fit column
        PROG_DISPLAY="${PROG_DISPLAY:0:22}"
        FULL_CMD_DISPLAY="${FULL_CMD:0:80}"

        printf "%-8s %-8s %-10s %-20s %-22s %-15s %-8s\n" \
            "$gpu_idx" "$pid" "$mem_used" "$AGE_STR" \
            "$PROG_DISPLAY" "${USER:-N/A}" "${CPU:-0.0}"
        printf "  CMD: %s\n\n" "${FULL_CMD_DISPLAY:-N/A}"

    done <<< "$GPU_PROCS"
fi

echo "=========================================="
echo "Timestamp: $(date)"