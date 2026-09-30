//! Idempotent initialization for legacy process-wide classifier slots.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug, PartialEq, Eq)]
struct Identity {
    directory: PathBuf,
    use_cpu: bool,
    max_sequence_length: usize,
}

/// Reuse an initialized classifier only for the same requested configuration.
/// Model directories must remain immutable for the process lifetime.
pub struct ClassifierSlot<T> {
    value: OnceLock<(Identity, Arc<T>)>,
    initialization: Mutex<()>,
}

impl<T> Default for ClassifierSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ClassifierSlot<T> {
    pub const fn new() -> Self {
        Self {
            value: OnceLock::new(),
            initialization: Mutex::new(()),
        }
    }

    pub fn get(&self) -> Option<&Arc<T>> {
        self.value.get().map(|(_, model)| model)
    }

    pub fn initialize<E: std::fmt::Display>(
        &self,
        directory: &str,
        use_cpu: bool,
        max_sequence_length: usize,
        load: impl FnOnce(&str, usize) -> Result<T, E>,
    ) -> Result<(), String> {
        let directory = std::fs::canonicalize(directory).map_err(|error| error.to_string())?;
        if !directory.is_dir() {
            return Err("Classifier model path must be a directory".into());
        }
        let identity = Identity {
            directory,
            use_cpu,
            max_sequence_length: if max_sequence_length == 0 {
                512
            } else {
                max_sequence_length
            },
        };
        let _guard = self
            .initialization
            .lock()
            .map_err(|_| "Classifier initialization lock was poisoned")?;
        if let Some((loaded, _)) = self.value.get() {
            return if loaded == &identity {
                Ok(())
            } else {
                Err("Classifier is already initialized with a different model, device, or context budget".into())
            };
        }
        let path = identity
            .directory
            .to_str()
            .ok_or("Classifier model path must be UTF-8")?;
        let model = load(path, identity.max_sequence_length).map_err(|error| error.to_string())?;
        self.value
            .set((identity, Arc::new(model)))
            .map_err(|_| "Classifier initialization changed while holding its lock".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

    struct Directory(PathBuf);

    impl Directory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "candle-classifier-slot-{}-{}",
                std::process::id(),
                NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }

    impl Drop for Directory {
        fn drop(&mut self) {
            std::fs::remove_dir(&self.0).unwrap();
        }
    }

    #[test]
    fn equivalent_paths_and_default_budget_reuse_one_model() {
        let directory = Directory::new();
        let slot = ClassifierSlot::new();
        slot.initialize(directory.path(), true, 0, |_, limit| {
            assert_eq!(limit, 512);
            Ok::<_, &str>(42)
        })
        .unwrap();
        slot.initialize(
            directory.0.join(".").to_str().unwrap(),
            true,
            512,
            |_, _| Err::<i32, _>("must not reload"),
        )
        .unwrap();
        assert_eq!(**slot.get().unwrap(), 42);
    }

    #[test]
    fn changed_model_device_or_budget_is_rejected_without_loading() {
        let first = Directory::new();
        let second = Directory::new();
        let slot = ClassifierSlot::new();
        slot.initialize(first.path(), true, 32768, |_, _| Ok::<_, &str>(42))
            .unwrap();
        for (path, cpu, limit) in [
            (second.path(), true, 32768),
            (first.path(), false, 32768),
            (first.path(), true, 512),
        ] {
            let result = slot.initialize(path, cpu, limit, |_, _| -> Result<i32, &str> {
                panic!("mismatched request must not load")
            });
            assert!(result.unwrap_err().contains("different"));
            assert_eq!(**slot.get().unwrap(), 42);
        }
    }

    #[test]
    fn failed_load_can_be_retried() {
        let directory = Directory::new();
        let slot = ClassifierSlot::new();
        assert!(slot
            .initialize(directory.path(), true, 512, |_, _| Err::<i32, _>("failure"))
            .is_err());
        assert!(slot.get().is_none());
        slot.initialize(directory.path(), true, 512, |_, _| Ok::<_, &str>(42))
            .unwrap();
        assert_eq!(**slot.get().unwrap(), 42);
    }

    #[test]
    fn concurrent_identical_initializers_load_once() {
        let directory = Directory::new();
        let slot = ClassifierSlot::new();
        let calls = AtomicUsize::new(0);
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    barrier.wait();
                    slot.initialize(directory.path(), true, 512, |_, _| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        Ok::<_, &str>(42)
                    })
                    .unwrap();
                });
            }
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(**slot.get().unwrap(), 42);
    }
}
