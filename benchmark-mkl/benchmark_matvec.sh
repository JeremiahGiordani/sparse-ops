#!/usr/bin/env bash
#
# benchmark/benchmark.sh
# — pin to cores, set thread‑affinity, warm up, then do the real run

# ---- Configurable params ----
M=2000
N=2000
SPARSITY=0.90
WARMUPS=5       # # of dummy runs to warm pages, threads, freq
RUNS=10
OMP_THREADS=8
MKL_THREADS=8
SEED=42
# ------------------------------

# 1) Export thread counts & affinity knobs
export OMP_NUM_THREADS=${OMP_THREADS}
export MKL_NUM_THREADS=${MKL_THREADS}
export MKL_DYNAMIC=FALSE
export KMP_BLOCKTIME=0

# 2) A little warm up to spin up threads, fault in pages, prime caches
echo "Warming up (${WARMUPS} runs)…"
for i in $(seq 1 $WARMUPS); do
  $CMD_PREFIX ./build/matvec/sparse_matvec_bench \
    --M ${M} --N ${N} --sparsity ${SPARSITY} --runs 1 --mkl-threads ${MKL_THREADS} --omp-threads ${OMP_THREADS} --seed ${SEED} \
    > /dev/null
done

# 3) The real benchmark
echo "Running real benchmark (${RUNS} runs)…"
$CMD_PREFIX ./build/matvec/sparse_matvec_bench \
  --M ${M} --N ${N} --sparsity ${SPARSITY} --runs ${RUNS} --mkl-threads ${MKL_THREADS} --omp-threads ${OMP_THREADS} --seed ${SEED} \
