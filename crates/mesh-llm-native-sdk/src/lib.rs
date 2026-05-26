//! Prebuilt native mesh-llm runtime.
//!
//! This crate's job is to *fetch and link* the matching `libmeshllm_ffi`
//! prebuilt static archive for the consumer's target platform and selected
//! backend. The archive contains patched llama.cpp, skippy, the mesh-llm
//! host runtime, and UniFFI-generated C ABI symbols. We expose a small
//! Rust API on top of those symbols.
//!
//! This is the same archive shape Swift consumes via `.binaryTarget`. The
//! difference is the consumer-side wrapper: Swift gets generated Swift
//! bindings; Rust gets the wrappers in this module.

// Force the linker to keep `libmeshllm_ffi` linked into the consumer's
// final binary. `build.rs` emits `cargo:rustc-link-search=...` so the
// linker can find the static archive; this `#[link]` attribute forces a
// `-l meshllm_ffi` even when the consumer hasn't yet referenced a symbol.
//
// `kind = "static"` matches the file the build script extracts on every
// platform — same shape as Swift's xcframework, which also ships a
// static archive that gets linked into the consumer app.
#[link(name = "meshllm_ffi", kind = "static")]
unsafe extern "C" {}

mod ffi;

pub use ffi::generate_owner_keypair_hex;
pub use ffi::uniffi_contract_version;
