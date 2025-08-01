#!/usr/bin/env bash
#
# benchmark/benchmark_matmul.sh
# — pin to cores, set thread‑affinity, warm up, then do the real run for sparse×dense matmul

# ---- Configurable params ----
M=2000
N=2000
C=120
SPARSITY=0.70
WARMUPS=5       # # of dummy runs to warm pages, threads, freq
RUNS=10
OMP_THREADS=8
OPENBLAS_THREADS=8
SEED=44
IRREGULAR=0    # set to 1 if you want the last row fully dense
# ------------------------------

# (Optional) Pin to cores 0–7; comment out if not available
# CMD_PREFIX="taskset -c 0-7"

# Export thread counts & affinity knobs
export OMP_NUM_THREADS=${OMP_THREADS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_THREADS}
# export OMP_PLACES=cores
# export OMP_PROC_BIND=spread
export OPENBLAS_BLOCKTIME=0 
# export KMP_BLOCKTIME=0
# export KMP_AFFINITY=granularity=fine,compact

echo "Warming up (${WARMUPS} runs)…"
for i in $(seq 1 $WARMUPS); do
  $CMD_PREFIX ./build/matmul/sparse_matmul_bench \
    --M ${M} --N ${N} --C ${C} \
    --sparsity ${SPARSITY} --runs 1 \
    --openblas-threads ${OPENBLAS_THREADS} --omp-threads ${OMP_THREADS} \
    --seed ${SEED} --irregular ${IRREGULAR} \
    > /dev/null
done

echo "Running real benchmark (${RUNS} runs)…"
$CMD_PREFIX ./build/matmul/sparse_matmul_bench \
  --M ${M} --N ${N} --C ${C} \
  --sparsity ${SPARSITY} --runs ${RUNS} \
  --openblas-threads ${OPENBLAS_THREADS} --omp-threads ${OMP_THREADS} \
  --seed ${SEED} --irregular ${IRREGULAR}
