use std::ffi::{c_char, CStr};

/// Parse a caller-owned array of C strings without extending its lifetime
/// beyond the enclosing FFI call.
///
/// # Safety
///
/// `values` must point to at least `length` readable C-string pointers. Each
/// non-null string must remain valid for the lifetime of the returned slice
/// references.
pub(super) unsafe fn parse_c_string_array<'a>(
    values: *const *const c_char,
    length: usize,
    field: &str,
) -> Result<Vec<&'a str>, String> {
    if values.is_null() {
        return Err(format!("{field} array is null"));
    }
    let mut parsed = Vec::with_capacity(length);
    for index in 0..length {
        let value = unsafe { *values.add(index) };
        if value.is_null() {
            return Err(format!("null {field} at index {index}"));
        }
        let text = unsafe { CStr::from_ptr(value) }
            .to_str()
            .map_err(|error| format!("invalid UTF-8 in {field} {index}: {error}"))?;
        parsed.push(text);
    }
    Ok(parsed)
}
