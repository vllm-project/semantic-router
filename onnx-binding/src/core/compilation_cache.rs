//! Per-content, per-shape immutable MIGraphX program storage.
//! Cold compilation holds the lock through its first actual inference.
//! Ready entries are copied under the lock, then used independently.
use super::artifact_identity::{json_digest, ArtifactDigest, ArtifactSnapshot};
use super::execution_contract::{validate_contract, ExecutionInput};
use super::migraphx_identity::GpuIdentity;
use super::unified_error::{errors, UnifiedResult};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize, Serializer};
use std::collections::BTreeMap;
#[cfg(target_os = "linux")]
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
#[cfg(target_os = "linux")]
use std::io::Read;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CompilationIdentity {
    pub version: u32,
    pub artifacts: Vec<ArtifactDigest>,
    pub inputs: Vec<ExecutionInput>,
    pub precision: String,
    pub runtime_build: String,
    pub runtime_artifacts: Vec<ArtifactDigest>,
    pub gpu: GpuIdentity,
    pub compiler_flags: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CompilationCacheEvidence {
    pub key: String,
    pub state: String,
    pub files: BTreeMap<String, String>,
    pub compiled_file_reads: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SharedCacheEvidence(pub Arc<Mutex<CompilationCacheEvidence>>);
impl Serialize for SharedCacheEvidence {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.lock().serialize(serializer)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadyEntry {
    identity: CompilationIdentity,
    files: BTreeMap<String, String>,
}

fn invalid(error: impl std::fmt::Display) -> super::unified_error::UnifiedError {
    errors::config_error("compilation_cache", &error.to_string())
}

fn unique_name(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "{prefix}-{}-{epoch}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn directory(path: &Path) -> anyhow::Result<()> {
    match std::fs::create_dir(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    existing_directory(path)
}

fn existing_directory(path: &Path) -> anyhow::Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    anyhow::ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "cache path is not a real directory"
    );
    Ok(())
}

fn program_files(path: &Path) -> anyhow::Result<BTreeMap<String, String>> {
    let mut files = BTreeMap::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("invalid cache filename"))?;
        anyhow::ensure!(
            !name.is_empty()
                && name.ends_with(".mxr")
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')),
            "unexpected compiled cache file"
        );
        let metadata = std::fs::symlink_metadata(entry.path())?;
        anyhow::ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink() && metadata.len() > 0,
            "invalid compiled program file"
        );
        files.insert(
            name,
            ArtifactSnapshot::capture(&entry.path(), "compiled-program")?
                .digest
                .sha256,
        );
    }
    anyhow::ensure!(
        !files.is_empty(),
        "provider produced no compiled cache files"
    );
    Ok(files)
}

fn copy_programs(
    source: &Path,
    destination: &Path,
    expected: &BTreeMap<String, String>,
) -> anyhow::Result<()> {
    for name in expected.keys() {
        // A private copy prevents provider writes from changing the ready entry.
        // Do not hardlink files which the provider may open for writing.
        std::fs::copy(source.join(name), destination.join(name))?;
        File::open(destination.join(name))?.sync_all()?;
    }
    anyhow::ensure!(
        &program_files(destination)? == expected,
        "compiled program changed during copy"
    );
    Ok(())
}

#[cfg(unix)]
fn lock_file(path: &Path) -> anyhow::Result<File> {
    use std::os::{fd::AsRawFd, unix::fs::OpenOptionsExt};
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    loop {
        // SAFETY: a live regular-file descriptor; the OS releases the lock on
        // close/process exit, including failed compilation.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } == 0 {
            break;
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error.into());
        }
    }
    Ok(file)
}
#[cfg(not(unix))]
fn lock_file(_path: &Path) -> anyhow::Result<File> {
    anyhow::bail!("compiled cache requires a process-shared filesystem lock")
}

struct ReadWatch {
    #[cfg(target_os = "linux")]
    file: File,
}
impl ReadWatch {
    fn new(path: &Path) -> anyhow::Result<Self> {
        #[cfg(target_os = "linux")]
        {
            use std::os::{fd::FromRawFd, unix::ffi::OsStrExt};
            let name = std::ffi::CString::new(path.as_os_str().as_bytes())?;
            let fd = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
            anyhow::ensure!(fd >= 0, "cache read observer initialization failed");
            let file = unsafe { File::from_raw_fd(fd) };
            let watch = unsafe {
                libc::inotify_add_watch(
                    fd,
                    name.as_ptr(),
                    libc::IN_ACCESS | libc::IN_MODIFY | libc::IN_CLOSE_WRITE,
                )
            };
            anyhow::ensure!(watch >= 0, "cache read observer installation failed");
            Ok(Self { file })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = path;
            Ok(Self {})
        }
    }
    #[cfg(not(target_os = "linux"))]
    fn finish(self) -> anyhow::Result<(Vec<String>, bool)> {
        Ok((Vec::new(), false))
    }
    #[cfg(target_os = "linux")]
    fn finish(mut self) -> anyhow::Result<(Vec<String>, bool)> {
        let mut names = BTreeSet::new();
        let mut modified = false;
        #[cfg(target_os = "linux")]
        {
            let mut buffer = [0_u8; 65536];
            loop {
                let count = match self.file.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(e) => return Err(e.into()),
                };
                let mut offset = 0;
                while offset < count {
                    anyhow::ensure!(
                        offset + std::mem::size_of::<libc::inotify_event>() <= count,
                        "truncated cache filesystem event"
                    );
                    let event = unsafe {
                        std::ptr::read_unaligned(
                            buffer.as_ptr().add(offset).cast::<libc::inotify_event>(),
                        )
                    };
                    offset += std::mem::size_of::<libc::inotify_event>();
                    anyhow::ensure!(
                        offset + event.len as usize <= count
                            && event.mask & libc::IN_Q_OVERFLOW == 0,
                        "cache filesystem observer overflow"
                    );
                    let bytes = &buffer[offset..offset + event.len as usize];
                    let end = bytes.iter().position(|&x| x == 0).unwrap_or(bytes.len());
                    let name = std::str::from_utf8(&bytes[..end])?;
                    if name.ends_with(".mxr") {
                        if event.mask & libc::IN_ACCESS != 0 {
                            names.insert(name.to_owned());
                        }
                        modified |= event.mask & (libc::IN_MODIFY | libc::IN_CLOSE_WRITE) != 0;
                    }
                    offset += event.len as usize;
                }
            }
        }
        Ok((names.into_iter().collect(), modified))
    }
}

pub struct CompilationCacheLease {
    pub evidence: SharedCacheEvidence,
    pub directory: PathBuf,
    identity: CompilationIdentity,
    parent: PathBuf,
    lock: Option<File>,
    watch: Option<ReadWatch>,
    ready_files: Option<BTreeMap<String, String>>,
    snapshots: Vec<ArtifactSnapshot>,
    completed: bool,
    failed: bool,
}

impl CompilationCacheLease {
    pub fn prepare(
        root: &Path,
        identity: CompilationIdentity,
        snapshots: Vec<ArtifactSnapshot>,
    ) -> UnifiedResult<Self> {
        validate_contract(&identity.inputs)?;
        let result = (|| -> anyhow::Result<Self> {
            std::fs::create_dir_all(root)?;
            directory(root)?;
            let root = root.canonicalize()?;
            let key = json_digest(&identity)?;
            let parent = root.join(&key);
            directory(&parent)?;
            let lock = lock_file(&parent.join("compile.lock"))?;
            let ready = parent.join("ready");
            let ready_files = if ready.exists() {
                existing_directory(&ready)?;
                let names = std::fs::read_dir(&ready)?
                    .map(|item| item.map(|entry| entry.file_name()))
                    .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
                anyhow::ensure!(
                    names
                        == ["entry.json", "programs"]
                            .map(std::ffi::OsString::from)
                            .into_iter()
                            .collect(),
                    "unexpected cache descriptor files"
                );
                let metadata = std::fs::symlink_metadata(ready.join("entry.json"))?;
                anyhow::ensure!(
                    metadata.is_file()
                        && !metadata.file_type().is_symlink()
                        && metadata.len() <= 4 * 1024 * 1024,
                    "invalid cache descriptor"
                );
                let entry: ReadyEntry =
                    serde_json::from_slice(&std::fs::read(ready.join("entry.json"))?)?;
                anyhow::ensure!(
                    entry.identity == identity,
                    "compiled cache descriptor differs"
                );
                let programs = ready.join("programs");
                existing_directory(&programs)?;
                anyhow::ensure!(
                    program_files(&programs)? == entry.files,
                    "compiled cache manifest differs"
                );
                Some(entry.files)
            } else {
                None
            };
            let work = parent.join(unique_name("work"));
            std::fs::create_dir(&work)?;
            if let Some(files) = &ready_files {
                copy_programs(&ready.join("programs"), &work, files)?;
            }
            for snapshot in &snapshots {
                snapshot.verify()?;
            }
            let watch = ReadWatch::new(&work)?;
            let evidence = SharedCacheEvidence(Arc::new(Mutex::new(CompilationCacheEvidence {
                key,
                state: if ready_files.is_some() {
                    "verified_entry_available"
                } else {
                    "prepared"
                }
                .into(),
                files: ready_files.clone().unwrap_or_default(),
                compiled_file_reads: Vec::new(),
            })));
            // A warm session owns a verified private copy and never publishes.
            // Releasing here lets other sessions prepare before this one runs.
            let lock = if ready_files.is_some() {
                drop(lock);
                None
            } else {
                Some(lock)
            };
            Ok(Self {
                evidence,
                directory: work,
                identity,
                parent,
                lock,
                watch: Some(watch),
                ready_files,
                snapshots,
                completed: false,
                failed: false,
            })
        })();
        result.map_err(invalid)
    }

    fn finish_first(&mut self) -> anyhow::Result<()> {
        // Stop observation before our own integrity reads, so they cannot be
        // mistaken for the provider actually consuming a compiled program.
        let (reads, modified) = self
            .watch
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing first-run observer"))?
            .finish()?;
        for snapshot in &self.snapshots {
            snapshot.verify()?;
        }
        let files = program_files(&self.directory)?;
        let state = if let Some(expected) = &self.ready_files {
            anyhow::ensure!(
                &files == expected && !modified,
                "provider rewrote a validated immutable cache entry"
            );
            if reads.iter().any(|name| files.contains_key(name)) {
                "reused"
            } else {
                "unconfirmed"
            }
        } else {
            let publish = self.parent.join(unique_name("publish"));
            std::fs::create_dir(&publish)?;
            let programs = publish.join("programs");
            std::fs::create_dir(&programs)?;
            copy_programs(&self.directory, &programs, &files)?;
            let mut descriptor = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(publish.join("entry.json"))?;
            descriptor.write_all(&serde_json::to_vec(&ReadyEntry {
                identity: self.identity.clone(),
                files: files.clone(),
            })?)?;
            descriptor.sync_all()?;
            File::open(&programs)?.sync_all()?;
            File::open(&publish)?.sync_all()?;
            anyhow::ensure!(
                !self.parent.join("ready").exists(),
                "ready entry appeared under exclusive lock"
            );
            std::fs::rename(&publish, self.parent.join("ready"))?;
            File::open(&self.parent)?.sync_all()?;
            "compiled"
        };
        let key = self.evidence.0.lock().key.clone();
        *self.evidence.0.lock() = CompilationCacheEvidence {
            key,
            state: state.into(),
            files,
            compiled_file_reads: reads,
        };
        self.completed = true;
        self.lock.take();
        Ok(())
    }
}

/// Guard only fixed cached execution. Dynamic CPU callers pass no lease.
pub fn with_inference<T>(
    lease: &mut Option<CompilationCacheLease>,
    actual: &[ExecutionInput],
    inference: impl FnOnce() -> UnifiedResult<T>,
) -> UnifiedResult<T> {
    let Some(lease) = lease else {
        return inference();
    };
    if lease.failed || actual != lease.identity.inputs {
        return Err(invalid(
            "cached session input contract changed or previous inference failed",
        ));
    }
    let result = inference();
    match result {
        Ok(value) => {
            if !lease.completed {
                if let Err(error) = lease.finish_first() {
                    lease.failed = true;
                    lease.evidence.0.lock().state = "failed".into();
                    lease.lock.take();
                    return Err(invalid(error));
                }
            }
            Ok(value)
        }
        Err(error) => {
            lease.failed = true;
            lease.evidence.0.lock().state = "failed".into();
            lease.lock.take();
            Err(error)
        }
    }
}
impl Drop for CompilationCacheLease {
    fn drop(&mut self) {
        // Failed stages are useful diagnostics and never eligible for reuse.
        // An unused warm session only owns a copy of an already verified entry;
        // closing it before its first inference must release that copy too.
        if !self.failed && (self.completed || self.ready_files.is_some()) {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> CompilationIdentity {
        CompilationIdentity {
            version: 1,
            artifacts: vec![ArtifactDigest {
                role: "graph".into(),
                sha256: "a".repeat(64),
            }],
            inputs: vec![ExecutionInput {
                name: "input_ids".into(),
                dtype: "int64".into(),
                shape: vec![1, 2048],
            }],
            precision: "native".into(),
            runtime_build: "test-runtime".into(),
            runtime_artifacts: vec![],
            gpu: GpuIdentity {
                pci_bdf: "0000:01:00.0".into(),
                uuid: "synthetic".into(),
                architectures: vec!["gfx942".into()],
                hip_runtime_version: 70000000,
                hip_driver_version: 70000000,
            },
            compiler_flags: [("exhaustive_tune".into(), "false".into())]
                .into_iter()
                .collect(),
        }
    }
    fn cold(root: &Path) -> CompilationCacheLease {
        let mut lease = Some(CompilationCacheLease::prepare(root, identity(), vec![]).unwrap());
        let work = lease.as_ref().unwrap().directory.clone();
        with_inference(&mut lease, &identity().inputs, || {
            std::fs::write(work.join("program.mxr"), b"compiled-data").unwrap();
            Ok(())
        })
        .unwrap();
        let lease = lease.unwrap();
        assert_eq!(lease.evidence.0.lock().state, "compiled");
        lease
    }
    #[test]
    fn cold_then_separate_warm_copy_preserves_ready_and_observes_reads() {
        let directory = tempfile::tempdir().unwrap();
        let first = cold(directory.path());
        let ready = first.parent.join("ready/programs/program.mxr");
        let evidence = first.evidence.clone();
        drop(first);
        assert!(ready.is_file());
        let mut warm =
            Some(CompilationCacheLease::prepare(directory.path(), identity(), vec![]).unwrap());
        assert_eq!(
            warm.as_ref().unwrap().evidence.0.lock().state,
            "verified_entry_available"
        );
        let path = warm.as_ref().unwrap().directory.join("program.mxr");
        assert_ne!(path, ready);
        with_inference(&mut warm, &identity().inputs, || {
            assert_eq!(std::fs::read(&path).unwrap(), b"compiled-data");
            Ok(())
        })
        .unwrap();
        #[cfg(target_os = "linux")]
        assert_eq!(warm.as_ref().unwrap().evidence.0.lock().state, "reused");
        #[cfg(not(target_os = "linux"))]
        assert_eq!(
            warm.as_ref().unwrap().evidence.0.lock().state,
            "unconfirmed"
        );
        assert_eq!(std::fs::read(&ready).unwrap(), b"compiled-data");
        assert_eq!(evidence.0.lock().state, "compiled");
        let mut changed = identity().inputs;
        changed[0].shape[1] = 1024;
        assert!(with_inference(&mut warm, &changed, || Ok(())).is_err());
    }
    #[test]
    fn failed_or_incomplete_compiles_are_not_published() {
        let directory = tempfile::tempdir().unwrap();
        let mut lease =
            Some(CompilationCacheLease::prepare(directory.path(), identity(), vec![]).unwrap());
        let parent = lease.as_ref().unwrap().parent.clone();
        let work = lease.as_ref().unwrap().directory.clone();
        assert!(with_inference(&mut lease, &identity().inputs, || Ok(())).is_err());
        assert!(!parent.join("ready").exists());
        assert_eq!(lease.as_ref().unwrap().evidence.0.lock().state, "failed");
        drop(lease);
        assert!(work.exists());
        let mut next =
            Some(CompilationCacheLease::prepare(directory.path(), identity(), vec![]).unwrap());
        assert!(
            with_inference(&mut next, &identity().inputs, || Err::<(), _>(invalid(
                "provider failed"
            )))
            .is_err()
        );
        assert!(!parent.join("ready").exists());
    }
    #[test]
    fn ready_sessions_can_prepare_before_either_runs() {
        let directory = tempfile::tempdir().unwrap();
        drop(cold(directory.path()));
        let first = CompilationCacheLease::prepare(directory.path(), identity(), vec![]).unwrap();
        let first_work = first.directory.clone();
        let root = directory.path().to_path_buf();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let other = std::thread::spawn(move || {
            let lease = CompilationCacheLease::prepare(&root, identity(), vec![]).unwrap();
            ready_tx.send(()).unwrap();
            lease
        });
        let prepared_without_first_inference = ready_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .is_ok();
        // Release the first lease before joining even on regression, so the
        // test reports the blocked preparation rather than hanging forever.
        drop(first);
        let second = other.join().unwrap();
        assert!(prepared_without_first_inference);
        assert!(
            !first_work.exists(),
            "closing an unused warm session must remove its private program copy"
        );
        assert_ne!(first_work, second.directory);
        let program = second.directory.join("program.mxr");
        let mut second = Some(second);
        with_inference(&mut second, &identity().inputs, || {
            assert_eq!(std::fs::read(&program).unwrap(), b"compiled-data");
            Ok(())
        })
        .unwrap();
    }
    #[test]
    fn corrupted_ready_entry_and_provider_rewrites_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let first = cold(directory.path());
        let parent = first.parent.clone();
        drop(first);
        let mut warm =
            Some(CompilationCacheLease::prepare(directory.path(), identity(), vec![]).unwrap());
        let path = warm.as_ref().unwrap().directory.join("program.mxr");
        assert!(with_inference(&mut warm, &identity().inputs, || {
            std::fs::write(&path, b"changed").unwrap();
            Ok(())
        })
        .is_err());
        assert_eq!(
            std::fs::read(parent.join("ready/programs/program.mxr")).unwrap(),
            b"compiled-data"
        );
        std::fs::write(parent.join("ready/programs/program.mxr"), b"corrupt").unwrap();
        assert!(CompilationCacheLease::prepare(directory.path(), identity(), vec![]).is_err());
    }
    #[test]
    fn first_use_is_serialized_until_publication_and_ready_is_never_partial() {
        let directory = tempfile::tempdir().unwrap();
        let mut first =
            Some(CompilationCacheLease::prepare(directory.path(), identity(), vec![]).unwrap());
        let work = first.as_ref().unwrap().directory.clone();
        let root = directory.path().to_path_buf();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let other = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let lease = CompilationCacheLease::prepare(&root, identity(), vec![]).unwrap();
            ready_tx
                .send(lease.evidence.0.lock().state.clone())
                .unwrap();
        });
        started_rx.recv().unwrap();
        assert!(ready_rx
            .recv_timeout(std::time::Duration::from_millis(25))
            .is_err());
        with_inference(&mut first, &identity().inputs, || {
            std::fs::write(work.join("program.mxr"), b"compiled-data").unwrap();
            Ok(())
        })
        .unwrap();
        assert_eq!(
            ready_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap(),
            "verified_entry_available"
        );
        other.join().unwrap();
    }

    #[test]
    fn content_precision_shape_and_compiler_identity_separate_entries() {
        let original = identity();
        let expected = json_digest(&original).unwrap();
        for field in 0..7 {
            let mut value = original.clone();
            match field {
                0 => value.artifacts[0].sha256 = "b".repeat(64),
                1 => value.inputs[0].shape[1] = 32768,
                2 => value.precision = "fp16".into(),
                3 => value.gpu.architectures = vec!["gfx950".into()],
                4 => value.runtime_build = "different".into(),
                5 => {
                    value
                        .compiler_flags
                        .insert("exhaustive_tune".into(), "true".into());
                }
                _ => value.artifacts.push(ArtifactDigest {
                    role: "external:weights.bin".into(),
                    sha256: "c".repeat(64),
                }),
            }
            assert_ne!(expected, json_digest(&value).unwrap());
        }
    }
    #[test]
    fn artifact_replacement_during_compilation_prevents_publication() {
        let directory = tempfile::tempdir().unwrap();
        let artifact = directory.path().join("graph.onnx");
        std::fs::write(&artifact, b"original").unwrap();
        let snapshot = ArtifactSnapshot::capture(&artifact, "graph").unwrap();
        let mut lease = Some(
            CompilationCacheLease::prepare(
                &directory.path().join("cache"),
                identity(),
                vec![snapshot],
            )
            .unwrap(),
        );
        let work = lease.as_ref().unwrap().directory.clone();
        let parent = lease.as_ref().unwrap().parent.clone();
        assert!(with_inference(&mut lease, &identity().inputs, || {
            std::fs::write(work.join("program.mxr"), b"compiled-data").unwrap();
            std::fs::write(&artifact, b"replacement-content").unwrap();
            Ok(())
        })
        .is_err());
        assert!(!parent.join("ready").exists());
    }
}
