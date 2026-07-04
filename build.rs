//! Link configuration for the optional `cblas-gemm` feature (see Cargo.toml):
//! resolves the system CBLAS that backs `server::CblasDgemm`.

fn main() {
    println!("cargo:rerun-if-env-changed=CBLAS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CBLAS_LIB_NAME");
    if std::env::var_os("CARGO_FEATURE_CBLAS_GEMM").is_none() {
        return;
    }

    // The server issues dgemm from many of its own threads at once, so the
    // linked BLAS must be safe under concurrent callers. Debian/Ubuntu's
    // *serial* OpenBLAS is NOT (its buffer pool is built without locking —
    // measured result corruption under 128 callers); the *pthread* build locks
    // the pool and, pinned to `OPENBLAS_NUM_THREADS=1`, matches serial
    // single-call speed. Probe the pthread directories; `CBLAS_LIB_DIR`
    // overrides. The chosen directory is also burned in as an rpath so the
    // runtime loader cannot silently substitute the unsafe serial alternative.
    let dir = std::env::var("CBLAS_LIB_DIR").ok().or_else(|| {
        [
            "/usr/lib/x86_64-linux-gnu/openblas-pthread", // Debian/Ubuntu
            "/usr/lib64/openblas-pthread",                // Fedora/RHEL
        ]
        .iter()
        .find(|d| std::path::Path::new(d).exists())
        .map(|d| d.to_string())
    });
    if let Some(dir) = dir {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
    let name = std::env::var("CBLAS_LIB_NAME").unwrap_or_else(|_| "openblas".to_string());
    println!("cargo:rustc-link-lib={name}");
    // When the linked BLAS is OpenBLAS, `CblasDgemm` pins the pool to one
    // thread per call via `openblas_set_num_threads` (the env var is read in
    // OpenBLAS's ELF constructor, before main can set it). Gated on a cfg so a
    // non-OpenBLAS `CBLAS_LIB_NAME` doesn't reference a missing symbol.
    println!("cargo:rustc-check-cfg=cfg(cblas_openblas)");
    if name.contains("openblas") {
        println!("cargo:rustc-cfg=cblas_openblas");
    }
}
