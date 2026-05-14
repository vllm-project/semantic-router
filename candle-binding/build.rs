fn main() {
    #[cfg(feature = "mkl")]
    {
        // Try MKLROOT first, then common conda paths
        let mkl_lib_dir = std::env::var("MKLROOT")
            .map(|root| format!("{}/lib", root))
            .or_else(|_| std::env::var("CONDA_PREFIX").map(|p| format!("{}/lib", p)))
            .unwrap_or_else(|_| "/usr/lib/x86_64-linux-gnu".to_string());

        println!("cargo:rustc-link-search=native={}", mkl_lib_dir);
        println!("cargo:rustc-link-lib=dylib=mkl_intel_lp64");
        println!("cargo:rustc-link-lib=dylib=mkl_sequential");
        println!("cargo:rustc-link-lib=dylib=mkl_core");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=m");
    }
}
