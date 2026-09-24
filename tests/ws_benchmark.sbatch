#!/bin/bash
#SBATCH --job-name=abiss-memory-gate
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss/build_mem/gate_scratch/benchmark_%j.log

set -eo pipefail
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKIO_WORKER_THREADS=4
ABISS=/projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss
export MPLCONFIGDIR=$ABISS/build_mem/gate_scratch/matplotlib
export TMPDIR=$ABISS/build_mem/gate_scratch
PYTHON_REPO=${1:?Pass the Python worktree path as the first argument}
python "$ABISS/tests/ws_gate.py" --mode benchmark --python-repo "$PYTHON_REPO"
