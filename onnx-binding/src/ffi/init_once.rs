//! Serialized, identity-aware publication for process-global native models.

use std::sync::{Mutex, OnceLock};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ModelInitIdentity {
    pub model_path: String,
    pub use_cpu: bool,
}

pub(super) struct InitializedModel<T> {
    pub identity: ModelInitIdentity,
    pub value: T,
}

pub(super) fn initialize_once_with_identity<T>(
    slot: &OnceLock<InitializedModel<T>>,
    init_lock: &Mutex<()>,
    identity: ModelInitIdentity,
    build: impl FnOnce() -> Result<T, String>,
) -> Result<(), String> {
    let _guard = init_lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    if let Some(existing) = slot.get() {
        return validate_identity(&existing.identity, &identity);
    }

    let candidate = InitializedModel {
        identity: identity.clone(),
        value: build()?,
    };
    match slot.set(candidate) {
        Ok(()) => Ok(()),
        Err(_) => slot
            .get()
            .ok_or_else(|| "global model publication lost without a winner".to_string())
            .and_then(|existing| validate_identity(&existing.identity, &identity)),
    }
}

fn validate_identity(
    existing: &ModelInitIdentity,
    requested: &ModelInitIdentity,
) -> Result<(), String> {
    if existing == requested {
        return Ok(());
    }
    Err(format!(
        "model is already initialized with a different path or device (loaded path={:?}, loaded use_cpu={}, requested path={:?}, requested use_cpu={})",
        existing.model_path, existing.use_cpu, requested.model_path, requested.use_cpu
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};

    fn identity(path: &str, use_cpu: bool) -> ModelInitIdentity {
        ModelInitIdentity {
            model_path: path.to_string(),
            use_cpu,
        }
    }

    #[test]
    fn exact_identity_is_idempotent_but_path_or_device_changes_fail() {
        let slot = OnceLock::new();
        let lock = Mutex::new(());
        initialize_once_with_identity(&slot, &lock, identity("model-a", true), || Ok(7))
            .expect("first initialization");
        initialize_once_with_identity(&slot, &lock, identity("model-a", true), || Ok(8))
            .expect("exact retry");
        assert_eq!(slot.get().expect("published").value, 7);
        assert!(
            initialize_once_with_identity(&slot, &lock, identity("model-b", true), || Ok(9))
                .is_err()
        );
        assert!(
            initialize_once_with_identity(&slot, &lock, identity("model-a", false), || Ok(10))
                .is_err()
        );
    }

    #[test]
    fn concurrent_exact_initializers_build_once_and_all_succeed() {
        let slot = Arc::new(OnceLock::new());
        let lock = Arc::new(Mutex::new(()));
        let barrier = Arc::new(Barrier::new(8));
        let builds = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let slot = Arc::clone(&slot);
            let lock = Arc::clone(&lock);
            let barrier = Arc::clone(&barrier);
            let builds = Arc::clone(&builds);
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                initialize_once_with_identity(&slot, &lock, identity("model-a", true), || {
                    builds.fetch_add(1, Ordering::SeqCst);
                    Ok(11)
                })
            }));
        }
        for worker in workers {
            worker.join().expect("worker").expect("initializer");
        }
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_eq!(slot.get().expect("published").value, 11);
    }

    #[test]
    fn failed_build_does_not_consume_the_slot() {
        let slot = OnceLock::new();
        let lock = Mutex::new(());
        assert!(
            initialize_once_with_identity(&slot, &lock, identity("model-a", true), || Err::<
                usize,
                _,
            >(
                "load failed".to_string()
            ))
            .is_err()
        );
        initialize_once_with_identity(&slot, &lock, identity("model-b", false), || Ok(12))
            .expect("retry after failure");
        assert_eq!(slot.get().expect("published").value, 12);
    }
}
