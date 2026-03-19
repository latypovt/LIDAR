#!/bin/bash
set -e # Terminate immediately on any internal failure

# Argument Validation
ANALYSIS_DIR=$1
MASK=$2

if [ -z "$ANALYSIS_DIR" ] || [ -z "$MASK" ]; then
    echo "ERROR: Missing arguments."
    echo "Usage: ./run_3dclustsim.sh /path/to/dbm/stats/analysis_name /path/to/brain_mask.nii.gz"
    exit 1
fi

# Determine directory of this script so we can find the Python companion
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "=========================================================="
echo " LIDAR: AFNI 3dClustSim & FWE Extraction Pipeline"
echo " Target: $ANALYSIS_DIR"
echo "=========================================================="

# Environment
module load afni/20191017 2>/dev/null || echo "Warning: module load afni failed. Assuming AFNI is in PATH."

# Target Files
RESIDUALS="${ANALYSIS_DIR}/4D_residuals.nii.gz"
ACF_FILE="${ANALYSIS_DIR}/acf_parameters.txt"
SIM_PREFIX="${ANALYSIS_DIR}/clust_sim_results"
SIM_OUT="${SIM_PREFIX}.NN1_bisided.1D"

if [ ! -f "$RESIDUALS" ]; then
    echo "FATAL ERROR: 4D_residuals.nii.gz not found in $ANALYSIS_DIR."
    exit 1
fi

# ---------------------------------------------------------
# STEP 1: Spatial Autocorrelation (Checkpoint)
# ---------------------------------------------------------
if [ -f "$ACF_FILE" ]; then
    echo "[CHECKPOINT] ACF parameters found. Skipping 3dFWHMx."
else
    echo "Calculating Spatial Autocorrelation (3dFWHMx)..."
    3dFWHMx -mask "$MASK" -input "$RESIDUALS" -acf > "$ACF_FILE"
fi

# ---------------------------------------------------------
# STEP 2: Monte Carlo Simulation (Checkpoint)
# ---------------------------------------------------------
if [ -f "$SIM_OUT" ]; then
    echo "[CHECKPOINT] ClustSim results found. Skipping 3dClustSim."
else
    echo "Running Monte Carlo Simulations (3dClustSim)..."
    ACF_PARAMS=$(tail -n 1 "$ACF_FILE" | awk '{print $1, $2, $3}')
    3dClustSim -mask "$MASK" -acf $ACF_PARAMS -athr 0.05 -pthr 0.01 0.005 0.001 -prefix "$SIM_PREFIX"
fi

# ---------------------------------------------------------
# STEP 3: Threshold Extraction
# ---------------------------------------------------------
echo "Extracting FWE Cluster Thresholds (alpha = 0.05)..."

# AFNI pivoted the table: Rows are p-values, Column 2 is the cluster size.
K_01=$(awk '$1 ~ /^0\.01/ {print $2}' "$SIM_OUT")
K_005=$(awk '$1 ~ /^0\.005/ {print $2}' "$SIM_OUT")
K_001=$(awk '$1 ~ /^0\.001/ {print $2}' "$SIM_OUT")

# FAILSAFE: If extraction fails, abort before crashing Python
if [ -z "$K_005" ]; then
    echo "FATAL ERROR: Could not parse k-thresholds. Raw AFNI output:"
    cat "$SIM_OUT"
    exit 1
fi

echo "  p < 0.01  requires k >= $K_01"
echo "  p < 0.005 requires k >= $K_005"
echo "  p < 0.001 requires k >= $K_001"
# ---------------------------------------------------------
# STEP 4: Execution Across All Model Features
# ---------------------------------------------------------
echo "Applying thresholds to all feature maps..."

# Loop through every subdirectory in the analysis folder
for FEAT_DIR in "$ANALYSIS_DIR"/*/; do
    if [ -f "${FEAT_DIR}map_punc.nii.gz" ]; then
        FEATURE_NAME=$(basename "$FEAT_DIR")
        echo "Processing feature: $FEATURE_NAME"
        
        python "${SCRIPT_DIR}/apply_fwe_thresh.py" "${FEAT_DIR}" 0.01 "$K_01"
        python "${SCRIPT_DIR}/apply_fwe_thresh.py" "${FEAT_DIR}" 0.005 "$K_005"
        python "${SCRIPT_DIR}/apply_fwe_thresh.py" "${FEAT_DIR}" 0.001 "$K_001"
    fi
done

echo "=========================================================="
echo " Pipeline Complete."
echo "=========================================================="