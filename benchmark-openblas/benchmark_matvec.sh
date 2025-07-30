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
OPENBLAS_THREADS=1
SEED=42
# ------------------------------

# 1) Pin to cores 0–7 (so threads never hop across sockets / NUMA)
#    and bind memory to node 0 (if NUMA-aware).
#    If you don't have numactl or can't, just use taskset below.
# CMD_PREFIX="taskset -c 0-7"
# CMD_PREFIX="numactl --physcpubind=0-7 --membind=0 taskset -c 0-7"

# 2) Export thread counts & affinity knobs
export OMP_NUM_THREADS=${OMP_THREADS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_THREADS}
export OPENBLAS_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export KMP_BLOCKTIME=0
# export KMP_AFFINITY="granularity=fine,compact"

# 3) A little warm up to spin up threads, fault in pages, prime caches
echo "Warming up (${WARMUPS} runs)…"
for i in $(seq 1 $WARMUPS); do
  $CMD_PREFIX ./build/matvec/sparse_matvec_bench \
    --M ${M} --N ${N} --sparsity ${SPARSITY} --runs 1 --openblas-threads ${OPENBLAS_THREADS} --omp-threads ${OMP_THREADS} --seed ${SEED} \
    > /dev/null
done

# 4) The real benchmark
echo "Running real benchmark (${RUNS} runs)…"
$CMD_PREFIX ./build/matvec/sparse_matvec_bench \
  --M ${M} --N ${N} --sparsity ${SPARSITY} --runs ${RUNS} --openblas-threads ${OPENBLAS_THREADS} --omp-threads ${OMP_THREADS} --seed ${SEED} \
