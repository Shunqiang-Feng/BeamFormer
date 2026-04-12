#!/bin/bash
# run_reproduce_figures.sh - Reproduce all figures from the BeamFormer paper artifact.
#
# Usage:
#   ./run_reproduce_figures.sh [--data_source cache|from_scratch] [--num_samples N]
#
# Options:
#   --data_source   'cache' (default) or 'from_scratch'
#   --num_samples   Limit test samples (only used with from_scratch)

set -e


DATA_SOURCE="cache"
NUM_SAMPLES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_source)
            DATA_SOURCE="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--data_source cache|from_scratch] [--num_samples N]"
            exit 1
            ;;
    esac
done

# Build common args
COMMON_ARGS="--data_source ${DATA_SOURCE}"
if [[ -n "$NUM_SAMPLES" ]]; then
    COMMON_ARGS="${COMMON_ARGS} --num_samples ${NUM_SAMPLES}"
fi

PYTHON="conda run -n accelerate python"
SCRIPT="reproduce_figures.py"

FIGURES=(
    "Overall performance of different approaches"
    "Impact of scene configurations"
    "Failure case under severe multi-paths"
    "Failure case under pure noise"
    "Beam Spectrum Resolution"
    "Multi Path Prediction Accuracy"
    "Comparison of model latencies"
    "Impact of latency and user mobility"
    "Comparison of positional encoders"
    "Comparison of reference beam settings"
    "Comparison of power estimators"
    "Impact of model parameters"
    "Performance with real-world data"
    "Example spectrum with Sivers SDR"
    "Example spectrum with IBM SDR"
    "UMAP Visualization of Simulation and Real-World Feature Distributions"
)

PASS=0
FAIL=0
FAILED_FIGURES=()
declare -A FIGURE_TIMES

TOTAL_START=$(date +%s)

echo "========================================"
echo " BeamFormer - Reproducing All Figures"
echo "========================================"
echo " data_source : ${DATA_SOURCE}"
if [[ -n "$NUM_SAMPLES" ]]; then
    echo " num_samples : ${NUM_SAMPLES}"
fi
echo "========================================"

for FIGURE in "${FIGURES[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Reproducing: ${FIGURE}"
    echo "----------------------------------------"
    FIG_START=$(date +%s)
    if $PYTHON $SCRIPT --figure "${FIGURE}" $COMMON_ARGS; then
        FIG_END=$(date +%s)
        FIG_ELAPSED=$((FIG_END - FIG_START))
        FIGURE_TIMES["${FIGURE}"]="${FIG_ELAPSED}s"
        echo "[OK] ${FIGURE} (${FIG_ELAPSED}s)"
        PASS=$((PASS + 1))
    else
        FIG_END=$(date +%s)
        FIG_ELAPSED=$((FIG_END - FIG_START))
        FIGURE_TIMES["${FIGURE}"]="${FIG_ELAPSED}s (FAILED)"
        echo "[WARNING] Failed: ${FIGURE} (${FIG_ELAPSED}s)"
        FAIL=$((FAIL + 1))
        FAILED_FIGURES+=("${FIGURE}")
    fi
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "========================================"
echo " Timing per Figure"
echo "========================================"
for FIGURE in "${FIGURES[@]}"; do
    printf "  %-50s %s\n" "${FIGURE}" "${FIGURE_TIMES[${FIGURE}]}"
done
echo "----------------------------------------"
printf "  %-50s %ds\n" "Total" "${TOTAL_ELAPSED}"
echo ""
echo "========================================"
echo " Summary"
echo "========================================"
echo " Passed : ${PASS}"
echo " Failed : ${FAIL}"
if [[ ${#FAILED_FIGURES[@]} -gt 0 ]]; then
    echo " Failed figures:"
    for F in "${FAILED_FIGURES[@]}"; do
        echo "   - ${F}"
    done
fi
echo "========================================"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
