pub mod asset;
pub mod source_ref;
pub mod store;
pub mod target;

pub use asset::{AssetMatchKind, PluginAsset, select_plugin_asset};
pub use source_ref::{GitHubPluginSource, PluginInstallRef, PluginVersion, parse_install_ref};
pub use store::{InstalledPluginMetadata, PluginStore};
pub use target::{ArchiveExt, PluginTarget, UnsupportedTarget};
