#[cfg(feature = "dynamic-native-runtime")]
mod dynamic {
    use anyhow::{Context, Result};
    use mesh_llm_native_runtime::{
        HostRuntimeProfile, NativeRuntimeCache, NativeRuntimeReleaseManifest, RuntimeSelection,
        select_native_runtime,
    };
    use mesh_llm_runtime_install::{
        NativeRuntimeInstallOptions, NativeRuntimeInstallStatus, current_skippy_abi_version,
        install_native_runtime,
    };
    use std::path::PathBuf;

    #[derive(Clone, Debug)]
    pub(crate) struct LoadedNativeRuntime {
        pub(crate) native_runtime_id: String,
        pub(crate) libraries: Vec<PathBuf>,
    }

    pub(crate) fn try_load_installed_native_runtime() -> Result<Option<LoadedNativeRuntime>> {
        if skippy_runtime::native_runtime_loaded() {
            return Ok(None);
        }
        let cache = default_native_runtime_cache()?;
        let installed = cache.installed()?;
        let skippy_abi = current_skippy_abi_version();
        let profile = host_runtime_profile();
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: crate::VERSION.to_string(),
            skippy_abi,
            artifacts: installed
                .iter()
                .map(|runtime| runtime.manifest.runtime.clone())
                .collect(),
        };
        let Some(candidate) = select_native_runtime(
            &manifest,
            &profile,
            crate::VERSION,
            &RuntimeSelection::Recommended,
        ) else {
            return Ok(None);
        };
        let installed = cache
            .find_installed(crate::VERSION, candidate.artifact.native_runtime_id())?
            .with_context(|| {
                format!(
                    "selected native runtime {} disappeared from the cache",
                    candidate.artifact.native_runtime_id()
                )
            })?;
        let plan = installed.load_plan()?;
        unsafe {
            skippy_runtime::load_native_runtime_libraries(&plan.libraries).with_context(|| {
                format!(
                    "load native runtime {} from {}",
                    plan.native_runtime_id,
                    plan.root.display()
                )
            })?;
        }
        Ok(Some(LoadedNativeRuntime {
            native_runtime_id: plan.native_runtime_id,
            libraries: plan.libraries,
        }))
    }

    pub(crate) async fn ensure_native_runtime_installed_and_loaded()
    -> Result<Option<LoadedNativeRuntime>> {
        if let Some(runtime) = try_load_installed_native_runtime()? {
            return Ok(Some(runtime));
        }
        if skippy_runtime::native_runtime_loaded() {
            return Ok(None);
        }

        let outcome = install_native_runtime(NativeRuntimeInstallOptions::default())
            .await
            .context("install MeshLLM native runtime from release manifest")?;
        match outcome.status {
            NativeRuntimeInstallStatus::AlreadyInstalled => {
                tracing::info!(
                    native_runtime_id = %outcome.runtime.native_runtime_id,
                    "MeshLLM native runtime was already installed"
                );
            }
            NativeRuntimeInstallStatus::Installed => {
                tracing::info!(
                    native_runtime_id = %outcome.runtime.native_runtime_id,
                    path = %outcome.runtime.path.display(),
                    "Installed MeshLLM native runtime"
                );
            }
        }

        try_load_installed_native_runtime()?
            .with_context(|| {
                format!(
                    "installed native runtime {} but could not load it",
                    outcome.runtime.native_runtime_id
                )
            })
            .map(Some)
    }

    fn default_native_runtime_cache() -> Result<NativeRuntimeCache> {
        crate::system::native_runtime_install::default_native_runtime_cache()
    }

    fn host_runtime_profile() -> HostRuntimeProfile {
        crate::system::native_runtime_install::host_runtime_profile()
    }
}

#[cfg(feature = "dynamic-native-runtime")]
pub(crate) use dynamic::*;

#[cfg(not(feature = "dynamic-native-runtime"))]
pub(crate) fn try_load_installed_native_runtime() -> anyhow::Result<Option<()>> {
    Ok(None)
}
