fn main() {
    #[cfg(feature = "mkl")]
    {
        // MKL: link the static mkl-static-lp64-iomp package from the ocipkg cache.
        // The package is downloaded automatically by intel-mkl-src's build script
        // (pulled in via candle-core's mkl feature) on first build, and extracted
        // to `$HOME/.local/share/ocipkg/ghcr.io/rust-math/rust-mkl/linux/
        // mkl-static-lp64-iomp/<tag>/`, where <tag> embeds the MKL version plus a
        // content hash (e.g. `__2020.1-3038006115`) that varies across releases
        // and machines. Do NOT hardcode <tag> (an earlier revision hardcoded
        // `__2020.1-3038006115`, which broke builds on any machine whose cache
        // held a different tag); discover it at build time instead. Any <tag>
        // directory containing all four archives linked below is a valid
        // candidate; when several exist the newest mtime wins, because
        // intel-mkl-src's build script has already run by the time this script
        // executes and downloaded the version pinned by Cargo.lock.
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let base = format!(
            "{}/.local/share/ocipkg/ghcr.io/rust-math/rust-mkl/linux/mkl-static-lp64-iomp",
            home
        );
        let entries = match std::fs::read_dir(&base) {
            Ok(entries) => entries,
            Err(err) => panic!(
                "mkl feature: cannot read ocipkg cache dir {base} ({err})\n\
                 hint: the static package is fetched by intel-mkl-src (via \
                 candle-core) on first build; build once with network access \
                 or pre-seed the ocipkg cache"
            ),
        };

        let mut candidates: Vec<(std::time::SystemTime, std::path::PathBuf)> = entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_dir()
                    && [
                        "libmkl_intel_lp64.a",
                        "libmkl_intel_thread.a",
                        "libmkl_core.a",
                        "libiomp5.a",
                    ]
                    .iter()
                    .all(|lib| path.join(lib).is_file())
            })
            .filter_map(|path| {
                let mtime = path.metadata().and_then(|m| m.modified()).ok()?;
                Some((mtime, path))
            })
            .collect();

        if candidates.is_empty() {
            panic!(
                "mkl feature: no ocipkg package directory containing all of \
                 libmkl_intel_lp64.a, libmkl_intel_thread.a, libmkl_core.a and \
                 libiomp5.a under {base}"
            );
        }
        // Newest first; ties broken by path for determinism.
        candidates.sort_by(|a, b| b.cmp(a));
        let (_, dir) = candidates.remove(0);
        if !candidates.is_empty() {
            println!(
                "cargo:warning=mkl: multiple ocipkg package versions found, using the newest: {}",
                dir.display()
            );
        }
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static=mkl_intel_lp64");
        println!("cargo:rustc-link-lib=static=mkl_intel_thread");
        println!("cargo:rustc-link-lib=static=mkl_core");
        println!("cargo:rustc-link-lib=static=iomp5");
    }
}
