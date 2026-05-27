use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;

use crate::target::ArchiveExt;

pub fn extract_plugin_archive(
    archive_path: &Path,
    archive_ext: ArchiveExt,
    plugin_name: &str,
    install_dir: &Path,
) -> Result<PathBuf> {
    let staging = tempfile::Builder::new()
        .prefix("mesh-plugin-extract-")
        .tempdir()
        .context("create plugin extract staging directory")?;

    match archive_ext {
        ArchiveExt::TarGz => extract_tar_gz(archive_path, staging.path())?,
        ArchiveExt::Zip => extract_zip(archive_path, staging.path())?,
    }

    let extracted_root = find_plugin_root(staging.path(), plugin_name)?;
    let final_dir = install_dir.join(plugin_name);
    if final_dir.exists() {
        fs::remove_dir_all(&final_dir)
            .with_context(|| format!("remove previous plugin install {}", final_dir.display()))?;
    }
    fs::create_dir_all(install_dir)
        .with_context(|| format!("create plugin install dir {}", install_dir.display()))?;
    fs::rename(&extracted_root, &final_dir)
        .or_else(|_| copy_dir_and_remove(&extracted_root, &final_dir))?;
    validate_installed_plugin(&final_dir, plugin_name)?;
    Ok(final_dir)
}

fn extract_tar_gz(archive_path: &Path, destination: &Path) -> Result<()> {
    let file = fs::File::open(archive_path)
        .with_context(|| format!("open plugin archive {}", archive_path.display()))?;
    let decoder = GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    archive
        .unpack(destination)
        .with_context(|| format!("extract plugin archive {}", archive_path.display()))?;
    Ok(())
}

fn extract_zip(archive_path: &Path, destination: &Path) -> Result<()> {
    let file = fs::File::open(archive_path)
        .with_context(|| format!("open plugin archive {}", archive_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("read plugin zip archive {}", archive_path.display()))?;
    for index in 0..archive.len() {
        let mut file = archive.by_index(index)?;
        let Some(enclosed) = file.enclosed_name() else {
            bail!("zip archive contains unsafe path: {}", file.name());
        };
        let output_path = destination.join(enclosed);
        if file.is_dir() {
            fs::create_dir_all(&output_path)
                .with_context(|| format!("create zip directory {}", output_path.display()))?;
        } else {
            if let Some(parent) = output_path.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create zip parent directory {}", parent.display()))?;
            }
            let mut output = fs::File::create(&output_path)
                .with_context(|| format!("create zip output {}", output_path.display()))?;
            std::io::copy(&mut file, &mut output)
                .with_context(|| format!("write zip output {}", output_path.display()))?;
        }
    }
    Ok(())
}

fn find_plugin_root(staging: &Path, plugin_name: &str) -> Result<PathBuf> {
    let expected = staging.join(plugin_name);
    if expected.join("plugin.toml").exists() {
        return Ok(expected);
    }

    let mut matches = Vec::new();
    for entry in
        fs::read_dir(staging).with_context(|| format!("read staging dir {}", staging.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir() && entry.path().join("plugin.toml").exists() {
            matches.push(entry.path());
        }
    }

    match matches.as_slice() {
        [path] => Ok(path.clone()),
        [] => bail!("plugin archive does not contain plugin.toml"),
        _ => bail!("plugin archive contains multiple plugin roots"),
    }
}

fn validate_installed_plugin(plugin_dir: &Path, plugin_name: &str) -> Result<()> {
    if !plugin_dir.join("plugin.toml").exists() {
        bail!("installed plugin is missing plugin.toml");
    }
    let executable = plugin_dir.join(format!("{plugin_name}{}", std::env::consts::EXE_SUFFIX));
    if !executable.exists() {
        bail!(
            "installed plugin is missing executable {}",
            executable.display()
        );
    }
    Ok(())
}

fn copy_dir_and_remove(from: &Path, to: &Path) -> Result<()> {
    copy_dir(from, to)?;
    fs::remove_dir_all(from)
        .with_context(|| format!("remove copied plugin source {}", from.display()))?;
    Ok(())
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to).with_context(|| format!("create directory {}", to.display()))?;
    for entry in fs::read_dir(from).with_context(|| format!("read directory {}", from.display()))? {
        let entry = entry?;
        let from_path = entry.path();
        let to_path = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&from_path, &to_path)?;
        } else {
            fs::copy(&from_path, &to_path).with_context(|| {
                format!("copy {} to {}", from_path.display(), to_path.display())
            })?;
        }
    }
    Ok(())
}
