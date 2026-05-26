//! Hand-written Rust wrappers around the UniFFI C ABI exported by the
//! prebuilt `libmeshllm_ffi` static archive.
//!
//! This module is the price of sharing one native artifact across all
//! language SDKs (Swift, Kotlin, Node, and us). UniFFI's bindgen does
//! not ship a Rust generator, so we wrap by hand. Keep this module
//! mechanical and small; consider switching to a generator if/when one
//! becomes available, or to a Rust-native artifact pipeline if this
//! grows too large.

#![allow(non_camel_case_types)]

use std::os::raw::c_char;
use std::slice;

/// Mirror of UniFFI's C ABI `RustBuffer`. The native archive owns the
/// memory; we read it then free it via `ffi_meshllm_ffi_rustbuffer_free`.
#[repr(C)]
#[derive(Copy, Clone)]
struct RustBuffer {
    capacity: u64,
    len: u64,
    data: *mut u8,
}

/// Mirror of UniFFI's C ABI `ForeignBytes` — caller-owned input bytes.
#[repr(C)]
#[derive(Copy, Clone)]
struct ForeignBytes {
    len: i32,
    data: *const u8,
}

/// Mirror of UniFFI's C ABI `RustCallStatus`. `code = 0` = success;
/// non-zero indicates an error whose details (UniFFI variant + message)
/// are encoded in `error_buf` as the function's declared error type.
#[repr(C)]
struct RustCallStatus {
    code: i8,
    error_buf: RustBuffer,
}

impl RustCallStatus {
    fn new() -> Self {
        Self {
            code: 0,
            error_buf: RustBuffer {
                capacity: 0,
                len: 0,
                data: std::ptr::null_mut(),
            },
        }
    }
}

unsafe extern "C" {
    fn ffi_meshllm_ffi_rustbuffer_alloc(size: u64, out_status: *mut RustCallStatus)
        -> RustBuffer;
    fn ffi_meshllm_ffi_rustbuffer_free(buf: RustBuffer, out_status: *mut RustCallStatus);
    fn ffi_meshllm_ffi_rustbuffer_from_bytes(
        bytes: ForeignBytes,
        out_status: *mut RustCallStatus,
    ) -> RustBuffer;

    fn ffi_meshllm_ffi_uniffi_contract_version() -> u32;

    fn uniffi_meshllm_ffi_fn_func_generate_owner_keypair_hex(
        out_status: *mut RustCallStatus,
    ) -> RustBuffer;
}

/// UniFFI contract version baked into the linked `libmeshllm_ffi`.
///
/// Useful as a sanity check after linking — must match the version this
/// wrapper crate was written against.
pub fn uniffi_contract_version() -> u32 {
    unsafe { ffi_meshllm_ffi_uniffi_contract_version() }
}

/// Generate a fresh hex-encoded owner keypair, as bytes that can be
/// passed to [`create_node`] or [`create_client`] later.
///
/// Runs the real mesh-llm key-generation code inside the linked native
/// archive — proof that the static archive's interior code is reachable
/// from a Rust consumer, not just the version-check symbol.
pub fn generate_owner_keypair_hex() -> String {
    let mut status = RustCallStatus::new();
    // Safety: signature matches the UniFFI ABI; on success the returned
    // RustBuffer owns memory we copy out and then free.
    let buf = unsafe {
        uniffi_meshllm_ffi_fn_func_generate_owner_keypair_hex(&mut status as *mut _)
    };
    assert_eq!(
        status.code, 0,
        "generate_owner_keypair_hex returned non-zero status {}",
        status.code,
    );
    let result = rust_buffer_into_string(buf);
    result
}

/// Take ownership of a UniFFI `RustBuffer` carrying UTF-8 string bytes,
/// copy it into a Rust `String`, then free the buffer via the native
/// archive's allocator.
fn rust_buffer_into_string(buf: RustBuffer) -> String {
    let s = if buf.data.is_null() || buf.len == 0 {
        String::new()
    } else {
        // Safety: native side guarantees `len` valid UTF-8 bytes at
        // `data` for an FfiConverterString lift.
        let slice = unsafe { slice::from_raw_parts(buf.data, buf.len as usize) };
        std::str::from_utf8(slice)
            .expect("native returned non-utf8 string")
            .to_string()
    };
    let mut free_status = RustCallStatus::new();
    // Safety: same buffer the native side handed us; free with the
    // matching allocator.
    unsafe { ffi_meshllm_ffi_rustbuffer_free(buf, &mut free_status as *mut _) };
    assert_eq!(
        free_status.code, 0,
        "rustbuffer_free returned non-zero status {}",
        free_status.code,
    );
    s
}

/// Suppress unused-import / unused-extern warnings for symbols we'll
/// need when wrapping the rest of the surface (creating nodes, etc.).
#[allow(dead_code, unused_unsafe)]
fn _keep_used() {
    let mut status = RustCallStatus::new();
    let _ = ForeignBytes {
        len: 0,
        data: std::ptr::null(),
    };
    let _ = c_char::default();
    let _ = unsafe { ffi_meshllm_ffi_rustbuffer_alloc(0, &mut status as *mut _) };
    let _ = unsafe {
        ffi_meshllm_ffi_rustbuffer_from_bytes(
            ForeignBytes {
                len: 0,
                data: std::ptr::null(),
            },
            &mut status as *mut _,
        )
    };
}
