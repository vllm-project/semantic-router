//! Runtime-selected GPU/compiler identity for the optional MIGraphX cache.
//! Uses stable scalar HIP and ROCr APIs; never guesses a hipDeviceProp_t layout.
use serde::{Deserialize, Serialize};

// MIGraphX loads its GPU target lazily. Bind these installed backend modules
// before cache lookup, then require the actual loaded closure to match after
// session creation. Merely hashing the libraries already mapped is incomplete.
#[cfg(any(target_os = "linux", test))]
fn gpu_backend_modules(core: &std::path::Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let core = core.canonicalize()?;
    let name = core.file_name().and_then(|value| value.to_str());
    let suffix = name
        .and_then(|value| value.strip_prefix("libmigraphx.so"))
        .ok_or_else(|| anyhow::anyhow!("unrecognized MIGraphX compiler filename"))?;
    let directory = core
        .parent()
        .ok_or_else(|| anyhow::anyhow!("MIGraphX compiler has no directory"))?;
    ["gpu", "device"]
        .iter()
        .map(|target| {
            let path = directory.join(format!("libmigraphx_{target}.so{suffix}"));
            anyhow::ensure!(path.is_file(), "MIGraphX GPU compiler module is missing");
            Ok(path.canonicalize()?)
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuIdentity {
    pub pci_bdf: String,
    pub uuid: String,
    pub architectures: Vec<String>,
    pub hip_runtime_version: i32,
    pub hip_driver_version: i32,
}

#[cfg(target_os = "linux")]
mod linux {
    use super::GpuIdentity;
    use crate::core::artifact_identity::ArtifactSnapshot;
    use std::ffi::{c_char, c_int, c_void, CStr, CString};
    use std::path::{Path, PathBuf};

    struct Library(*mut c_void);
    impl Library {
        fn open(path: &str) -> anyhow::Result<Self> {
            let path = CString::new(path)?;
            // SAFETY: a terminated path and standard loader flags; the handle
            // remains alive until all scalar API calls using its symbols end.
            let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            anyhow::ensure!(
                !handle.is_null(),
                "required ROCm runtime could not be loaded"
            );
            Ok(Self(handle))
        }
        unsafe fn symbol<T: Copy>(&self, name: &CStr) -> anyhow::Result<T> {
            let pointer = libc::dlsym(self.0, name.as_ptr());
            anyhow::ensure!(
                !pointer.is_null() && std::mem::size_of::<T>() == std::mem::size_of_val(&pointer),
                "required ROCm scalar API unavailable"
            );
            Ok(std::mem::transmute_copy(&pointer))
        }
    }
    impl Drop for Library {
        fn drop(&mut self) {
            unsafe { libc::dlclose(self.0) };
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Handle {
        value: u64,
    }
    type GetInfo = unsafe extern "C" fn(Handle, u32, *mut c_void) -> u32;
    type Iterate =
        unsafe extern "C" fn(unsafe extern "C" fn(Handle, *mut c_void) -> u32, *mut c_void) -> u32;
    type IterateIsas = unsafe extern "C" fn(
        Handle,
        unsafe extern "C" fn(Handle, *mut c_void) -> u32,
        *mut c_void,
    ) -> u32;
    struct AgentQuery {
        agent_info: GetInfo,
        isa_info: GetInfo,
        iterate_isas: IterateIsas,
        domain: u32,
        location: u32,
        matches: usize,
        uuid: Option<String>,
        architectures: Vec<String>,
        failed: bool,
    }

    unsafe extern "C" fn isa_callback(isa: Handle, context: *mut c_void) -> u32 {
        let query = &mut *context.cast::<AgentQuery>();
        let mut length = 0_u32;
        if (query.isa_info)(isa, 0, (&mut length as *mut u32).cast()) != 0
            || length == 0
            || length > 1024
        {
            query.failed = true;
            return 1;
        }
        let mut name = vec![0_u8; length as usize + 1];
        if (query.isa_info)(isa, 1, name.as_mut_ptr().cast()) != 0 {
            query.failed = true;
            return 1;
        }
        let end = name
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(name.len());
        match std::str::from_utf8(&name[..end]) {
            Ok(value) if value.starts_with("amdgcn-amd-amdhsa--gfx") => {
                query.architectures.push(value.to_string())
            }
            _ => {
                query.failed = true;
                return 1;
            }
        }
        0
    }

    unsafe extern "C" fn agent_callback(agent: Handle, context: *mut c_void) -> u32 {
        let query = &mut *context.cast::<AgentQuery>();
        let mut device = 0_u32;
        // Public ROCr header: HSA_AGENT_INFO_DEVICE=17, HSA_DEVICE_TYPE_GPU=1.
        if (query.agent_info)(agent, 17, (&mut device as *mut u32).cast()) != 0 {
            query.failed = true;
            return 1;
        }
        if device != 1 {
            return 0;
        }
        let (mut domain, mut location) = (0_u32, 0_u32);
        // ROCr hsa_ext_amd.h scalar DOMAIN/BDFID attributes, not ordinals.
        if (query.agent_info)(agent, 0xA00F, (&mut domain as *mut u32).cast()) != 0
            || (query.agent_info)(agent, 0xA006, (&mut location as *mut u32).cast()) != 0
        {
            query.failed = true;
            return 1;
        }
        if domain != query.domain || location != query.location {
            return 0;
        }
        query.matches += 1;
        let mut uuid = [0_u8; 21];
        if (query.agent_info)(agent, 0xA011, uuid.as_mut_ptr().cast()) != 0 {
            query.failed = true;
            return 1;
        }
        let Some(end) = uuid.iter().position(|&byte| byte == 0) else {
            query.failed = true;
            return 1;
        };
        match std::str::from_utf8(&uuid[..end]) {
            Ok(value)
                if value.starts_with("GPU-")
                    && value.len() == 20
                    && value[4..].bytes().all(|byte| byte.is_ascii_hexdigit()) =>
            {
                query.uuid = Some(value.to_owned())
            }
            _ => {
                query.failed = true;
                return 1;
            }
        }
        let result = (query.iterate_isas)(agent, isa_callback, context);
        if result != 0 {
            query.failed = true;
        }
        result
    }

    fn pci_location(text: &str) -> anyhow::Result<(u32, u32)> {
        let parts: Vec<_> = text.split([':', '.']).collect();
        anyhow::ensure!(parts.len() == 4, "invalid HIP PCI bus identity");
        let values: Vec<u32> = parts
            .iter()
            .map(|value| u32::from_str_radix(value, 16))
            .collect::<Result<_, _>>()?;
        anyhow::ensure!(
            values[0] <= 0xffff && values[1] <= 0xff && values[2] <= 0x1f && values[3] <= 7,
            "invalid HIP PCI bus components"
        );
        Ok((values[0], (values[1] << 8) | (values[2] << 3) | values[3]))
    }

    pub fn gpu_identity(device: i32) -> anyhow::Result<GpuIdentity> {
        anyhow::ensure!(device >= 0, "GPU device ordinal must be nonnegative");
        let hip = Library::open("libamdhip64.so")?;
        let rocr = Library::open("libhsa-runtime64.so.1")?;
        type Pci = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
        type Version = unsafe extern "C" fn(*mut c_int) -> c_int;
        type Initialize = unsafe extern "C" fn() -> u32;
        // Each signature is the scalar public HIP/ROCr C declaration. All
        // library handles remain in scope through callbacks and shutdown.
        let (
            pci,
            runtime,
            driver,
            initialize,
            shutdown,
            agent_info,
            isa_info,
            iterate,
            iterate_isas,
        ) = unsafe {
            (
                hip.symbol::<Pci>(c"hipDeviceGetPCIBusId")?,
                hip.symbol::<Version>(c"hipRuntimeGetVersion")?,
                hip.symbol::<Version>(c"hipDriverGetVersion")?,
                rocr.symbol::<Initialize>(c"hsa_init")?,
                rocr.symbol::<Initialize>(c"hsa_shut_down")?,
                rocr.symbol::<GetInfo>(c"hsa_agent_get_info")?,
                rocr.symbol::<GetInfo>(c"hsa_isa_get_info_alt")?,
                rocr.symbol::<Iterate>(c"hsa_iterate_agents")?,
                rocr.symbol::<IterateIsas>(c"hsa_agent_iterate_isas")?,
            )
        };
        let mut bus = [0 as c_char; 32];
        let (mut runtime_version, mut driver_version) = (0, 0);
        anyhow::ensure!(
            unsafe { pci(bus.as_mut_ptr(), bus.len() as i32, device) } == 0,
            "HIP device PCI identity unavailable"
        );
        anyhow::ensure!(bus.contains(&0), "unterminated HIP PCI identity");
        let bus = unsafe { CStr::from_ptr(bus.as_ptr()) }.to_str()?.to_owned();
        let (domain, location) = pci_location(&bus)?;
        anyhow::ensure!(
            unsafe { runtime(&mut runtime_version) } == 0
                && unsafe { driver(&mut driver_version) } == 0,
            "HIP runtime versions unavailable"
        );
        anyhow::ensure!(unsafe { initialize() } == 0, "ROCr initialization failed");
        let mut query = AgentQuery {
            agent_info,
            isa_info,
            iterate_isas,
            domain,
            location,
            matches: 0,
            uuid: None,
            architectures: Vec::new(),
            failed: false,
        };
        let result = unsafe { iterate(agent_callback, (&mut query as *mut AgentQuery).cast()) };
        // Balance only this call's reference; never reset the shared GPU.
        let stopped = unsafe { shutdown() };
        anyhow::ensure!(
            result == 0
                && stopped == 0
                && !query.failed
                && query.matches == 1
                && !query.architectures.is_empty(),
            "selected HIP device does not have one verified ROCr architecture identity"
        );
        query.architectures.sort();
        query.architectures.dedup();
        Ok(GpuIdentity {
            pci_bdf: bus,
            uuid: query
                .uuid
                .ok_or_else(|| anyhow::anyhow!("missing GPU UUID"))?,
            architectures: query.architectures,
            hip_runtime_version: runtime_version,
            hip_driver_version: driver_version,
        })
    }

    pub fn runtime_artifacts() -> anyhow::Result<Vec<ArtifactSnapshot>> {
        let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
        let function = ort::api().GetBuildInfoString as *const () as *const c_void;
        anyhow::ensure!(
            unsafe { libc::dladdr(function, &mut info) } != 0 && !info.dli_fname.is_null(),
            "actual ORT runtime location unavailable"
        );
        let runtime = PathBuf::from(unsafe { CStr::from_ptr(info.dli_fname) }.to_str()?);
        let provider = runtime
            .parent()
            .ok_or_else(|| anyhow::anyhow!("runtime has no directory"))?
            .join("libonnxruntime_providers_migraphx.so");
        let mut selected = std::collections::BTreeMap::new();
        selected.insert("ort-runtime".to_string(), runtime);
        selected.insert("migraphx-provider".to_string(), provider.clone());
        let mapped = std::fs::read_to_string("/proc/self/maps")?;
        let mut provider_loaded = false;
        for line in mapped.lines() {
            let Some(offset) = line.find('/') else {
                continue;
            };
            let path = Path::new(&line[offset..]);
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name == "libonnxruntime_providers_migraphx.so" {
                anyhow::ensure!(
                    path.canonicalize()? == provider.canonicalize()?,
                    "loaded MIGraphX provider is outside actual ORT runtime"
                );
                provider_loaded = true;
            }
            if name.starts_with("libmigraphx")
                || name.starts_with("libamdhip64.so")
                || name.starts_with("libhsa-runtime64.so")
            {
                let role = format!("compiler:{name}");
                if let Some(old) = selected.insert(role, path.to_path_buf()) {
                    anyhow::ensure!(
                        old.canonicalize()? == path.canonicalize()?,
                        "multiple compiler libraries with the same name"
                    );
                }
            }
        }
        anyhow::ensure!(
            provider_loaded
                && selected
                    .keys()
                    .any(|name| name.starts_with("compiler:libmigraphx")),
            "actual MIGraphX compiler libraries have not loaded"
        );
        let cores: Vec<_> = selected
            .iter()
            .filter(|(role, _)| role.starts_with("compiler:libmigraphx.so"))
            .map(|(_, path)| path.clone())
            .collect();
        anyhow::ensure!(
            cores.len() == 1,
            "expected one actual MIGraphX core library"
        );
        for path in super::gpu_backend_modules(&cores[0])? {
            let name = path.file_name().and_then(|name| name.to_str()).unwrap();
            let role = format!("compiler:{name}");
            if let Some(loaded) = selected.insert(role, path.clone()) {
                anyhow::ensure!(
                    loaded.canonicalize()? == path,
                    "loaded MIGraphX backend differs from installed compiler closure"
                );
            }
        }
        selected
            .into_iter()
            .map(|(role, path)| ArtifactSnapshot::capture(&path, role))
            .collect()
    }
}

#[cfg(target_os = "linux")]
pub use linux::{gpu_identity, runtime_artifacts};

#[cfg(not(target_os = "linux"))]
pub fn gpu_identity(_device: i32) -> anyhow::Result<GpuIdentity> {
    anyhow::bail!("MIGraphX compiled caching requires verified Linux ROCm runtime identity")
}

#[cfg(not(target_os = "linux"))]
pub fn runtime_artifacts() -> anyhow::Result<Vec<crate::core::artifact_identity::ArtifactSnapshot>>
{
    anyhow::bail!("MIGraphX compiled caching requires actual Linux runtime library identities")
}

#[cfg(test)]
mod tests {
    #[test]
    fn lazy_backend_identity_requires_both_version_matched_modules() {
        let directory = tempfile::tempdir().unwrap();
        let core = directory.path().join("libmigraphx.so.123");
        std::fs::write(&core, b"core").unwrap();
        assert!(super::gpu_backend_modules(&core).is_err());
        let gpu = directory.path().join("libmigraphx_gpu.so.123");
        std::fs::write(&gpu, b"gpu").unwrap();
        assert!(super::gpu_backend_modules(&core).is_err());
        let device = directory.path().join("libmigraphx_device.so.123");
        std::fs::write(&device, b"device").unwrap();
        assert_eq!(
            super::gpu_backend_modules(&core).unwrap(),
            vec![gpu.canonicalize().unwrap(), device.canonicalize().unwrap()]
        );
    }
}
