#!/usr/bin/env bash
#SBATCH --job-name=aospectrum-band
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

source /path/to/aospectrum-environment/bin/activate
AOSPECTRUM_GPU_COUNT="${AOSPECTRUM_GPU_COUNT:-8}"
THREADS_PER_WORKER=$((SLURM_CPUS_PER_TASK / AOSPECTRUM_GPU_COUNT))
if (( THREADS_PER_WORKER < 1 )); then
  THREADS_PER_WORKER=1
fi
export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${THREADS_PER_WORKER}"

# The Band TOML should list logical devices [0, 1, ..., 7].
srun aospectrum band band.toml
# To continue the exact interrupted run:
# srun aospectrum band band.toml --resume
