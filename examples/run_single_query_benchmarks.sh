#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

if ! command -v cargo >/dev/null 2>&1; then
    printf 'error: cargo is required\n' >&2
    exit 1
fi

if ! command -v nproc >/dev/null 2>&1; then
    printf 'error: nproc is required to select the maximum worker count\n' >&2
    exit 1
fi

if [[ -r /proc/cpuinfo ]]; then
    for feature in avx512f avx512dq; do
        if ! grep -qw "${feature}" /proc/cpuinfo; then
            printf 'error: CPU feature %s is required by this benchmark configuration\n' \
                "${feature}" >&2
            exit 1
        fi
    done
fi

max_threads="${PIR_BENCH_MAX_THREADS:-$(nproc)}"
if [[ ! "${max_threads}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'error: PIR_BENCH_MAX_THREADS must be a positive integer, got %q\n' \
        "${max_threads}" >&2
    exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
results_root="${PIR_BENCH_RESULTS_DIR:-${repo_root}/benchmark-results}"
results_dir="${results_root%/}/single-query-${timestamp}"
mkdir -p "${results_dir}"

features="avx512-fhe,cblas-gemm,numa-db-interleave"
default_rustflags="-C target-cpu=native -C target-feature=+avx512f,+avx512dq"
export RUSTFLAGS="${PIR_BENCH_RUSTFLAGS:-${RUSTFLAGS:-${default_rustflags}}}"

# Poulpy-PIR owns the parallelism. OpenBLAS must remain single-threaded inside
# each concurrently issued dgemm call.
export OPENBLAS_NUM_THREADS=1
export PIR_THREADS="${max_threads}"
export PIR_SETUP_THREADS="${max_threads}"
export PIR_OFFLINE_THREADS="${max_threads}"
export PIR_RESP2_SCRATCH=pooled

# Use the response-2 scheduler's budget-aware defaults for every online width;
# do not inherit a one-off schedule from the caller's shell.
unset PIR_RESP2_OUTER_THREADS
unset PIR_RESP2_INNER_THREADS

{
    printf 'started_utc=%s\n' "${timestamp}"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'max_threads=%s\n' "${max_threads}"
    printf 'features=%s\n' "${features}"
    printf 'rustflags=%s\n' "${RUSTFLAGS}"
    printf 'openblas_num_threads=%s\n' "${OPENBLAS_NUM_THREADS}"
    printf 'pir_resp2_scratch=%s\n' "${PIR_RESP2_SCRATCH}"
    if command -v lscpu >/dev/null 2>&1; then
        printf '\n'
        lscpu
    fi
} >"${results_dir}/machine.txt"

printf 'Building the optimized example once...\n'
cargo build --locked --release --features "${features}" --example pir

printf 'suite\tdatabase\tpreset\tonline_threads\tlog\n' \
    >"${results_dir}/manifest.tsv"

run_case() {
    local suite="$1"
    local database="$2"
    local preset="$3"
    local online_threads="$4"
    local log_name="${suite}_${database}_online-${online_threads}.log"
    local log_path="${results_dir}/${log_name}"

    printf '\nRunning suite=%s database=%s online_threads=%s\n' \
        "${suite}" "${database}" "${online_threads}"
    printf 'Log: %s\n' "${log_path}"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${suite}" "${database}" "${preset}" "${online_threads}" "${log_name}" \
        >>"${results_dir}/manifest.tsv"

    {
        printf 'suite                         : %s\n' "${suite}"
        printf 'database                      : %s\n' "${database}"
        printf 'PIR_ONLINE_THREADS            : %s\n' "${online_threads}"
        printf 'PIR_OFFLINE_THREADS           : %s\n' "${PIR_OFFLINE_THREADS}"
        printf 'PIR_SETUP_THREADS             : %s\n' "${PIR_SETUP_THREADS}"
        PIR_ONLINE_THREADS="${online_threads}" \
            cargo run --locked --quiet --release \
                --features "${features}" \
                --example pir -- "${preset}" 1
    } 2>&1 | tee "${log_path}"
}

# Matrix 1: database-size scaling at one online worker.
run_case size-sweep 32GiB InsPIRe2-g32-32GiB-c262144 1
run_case size-sweep 16GiB InsPIRe2-g32-16GiB-c262144 1
run_case size-sweep 8GiB InsPIRe2-g32-8GiB-c131072 1
run_case size-sweep 4GiB InsPIRe2-g32-4GiB-c131072 1
run_case size-sweep 2GiB InsPIRe2-g32-2GiB-c65536 1
run_case size-sweep 1GiB InsPIRe2-g32-1GiB-c32768 1

# Matrix 2: online scaling for the 32 GiB database. The one-thread case is
# intentionally repeated so this suite is independently comparable.
for online_threads in 1 2 4 8 16 32 64; do
    run_case thread-sweep 32GiB InsPIRe2-g32-32GiB-c262144 "${online_threads}"
done

printf '\nAll benchmark cases completed. Results: %s\n' "${results_dir}"
