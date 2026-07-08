//! MIGraphX execution-provider registration helpers.

use ort::ortsys;
use ort::session::builder::SessionBuilder;
use ort::AsPointer;
use std::ffi::CString;

/// Register MIGraphX through ONNX Runtime's generic provider-options API.
///
/// The typed `OrtMIGraphXProviderOptions` layout changed across ORT releases.
/// The generic string-key API matches ONNX Runtime's Python provider path and
/// avoids ABI mismatches when dynamically loading AMD ORT 1.23.x.
pub fn append_migraphx_execution_provider(
    builder: &mut SessionBuilder,
    device_id: i32,
) -> Result<(), ort::Error> {
    let provider = CString::new("MIGraphXExecutionProvider").expect("static provider name");
    let device_key = CString::new("device_id").expect("static option key");
    let device_value = CString::new(device_id.to_string()).expect("device id string");

    let keys = [device_key.as_ptr()];
    let values = [device_value.as_ptr()];

    ort::ortsys![unsafe SessionOptionsAppendExecutionProvider(
        builder.ptr_mut(),
        provider.as_ptr(),
        keys.as_ptr(),
        values.as_ptr(),
        keys.len(),
    )?];

    Ok(())
}
