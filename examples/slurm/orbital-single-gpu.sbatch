#!/usr/bin/env bash
#SBATCH --job-name=aospectrum-orbital
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

source /path/to/aospectrum-environment/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# The Orbital TOML should use device = 0.
srun aospectrum orbital orbital.toml
# To continue the exact interrupted eigensystem and state fields:
# srun aospectrum orbital orbital.toml --resume
