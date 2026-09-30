//! Stream the ONNX envelope, skipping tensor bytes, to bind every external data file.
use super::artifact_identity::ArtifactSnapshot;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Component, Path};

pub fn capture_onnx(path: &Path) -> anyhow::Result<Vec<ArtifactSnapshot>> {
    let graph = ArtifactSnapshot::capture(path, "graph")?;
    let mut locations = BTreeSet::new();
    walk(
        &mut BufReader::new(File::open(path)?),
        "model",
        0,
        &mut locations,
    )?;
    let mut artifacts = vec![graph];
    for location in locations {
        let relative = Path::new(&location);
        anyhow::ensure!(
            !location.is_empty()
                && !location.contains('\\')
                && relative
                    .components()
                    .all(|part| matches!(part, Component::Normal(_))),
            "unsafe ONNX external data location"
        );
        artifacts.push(ArtifactSnapshot::capture(
            &path.parent().unwrap_or(Path::new(".")).join(relative),
            format!("external:{location}"),
        )?);
    }
    Ok(artifacts)
}

fn varint(reader: &mut dyn Read) -> anyhow::Result<Option<u64>> {
    let mut value = 0;
    for shift in (0..70).step_by(7) {
        let mut byte = [0];
        match reader.read(&mut byte)? {
            0 if shift == 0 => return Ok(None),
            0 => anyhow::bail!("truncated ONNX varint"),
            _ => {}
        }
        anyhow::ensure!(shift != 63 || byte[0] <= 1, "overflowed ONNX varint");
        value |= u64::from(byte[0] & 127) << shift;
        if byte[0] < 128 {
            return Ok(Some(value));
        }
    }
    anyhow::bail!("invalid ONNX varint")
}

fn required_varint(reader: &mut dyn Read) -> anyhow::Result<u64> {
    varint(reader)?.ok_or_else(|| anyhow::anyhow!("missing ONNX varint"))
}

fn discard(reader: &mut dyn Read, size: u64) -> anyhow::Result<()> {
    let copied = std::io::copy(&mut reader.take(size), &mut std::io::sink())?;
    anyhow::ensure!(copied == size, "truncated ONNX field");
    Ok(())
}

fn child(kind: &str, field: u64) -> Option<&'static str> {
    match (kind, field) {
        ("model", 7) => Some("graph"),
        ("model", 25) => Some("function"),
        ("graph", 1) | ("function", 7) => Some("node"),
        ("graph", 5) | ("sparse", 1 | 2) | ("attribute", 5 | 10) => Some("tensor"),
        ("graph", 15) | ("attribute", 22 | 23) => Some("sparse"),
        ("node", 5) | ("function", 11) => Some("attribute"),
        ("attribute", 6 | 11) => Some("graph"),
        _ => None,
    }
}

fn external_entry(reader: &mut dyn Read) -> anyhow::Result<(String, String)> {
    let mut key = String::new();
    let mut value = String::new();
    while let Some(tag) = varint(reader)? {
        anyhow::ensure!(
            tag & 7 == 2 && matches!(tag >> 3, 1 | 2),
            "invalid ONNX external data entry"
        );
        let size = required_varint(reader)?;
        anyhow::ensure!(size <= 1 << 20, "oversize ONNX external data string");
        let mut data = vec![0; size as usize];
        reader.read_exact(&mut data)?;
        let text = String::from_utf8(data)?;
        if tag >> 3 == 1 {
            key = text
        } else {
            value = text
        }
    }
    Ok((key, value))
}

fn walk(
    reader: &mut dyn Read,
    kind: &str,
    depth: usize,
    locations: &mut BTreeSet<String>,
) -> anyhow::Result<()> {
    anyhow::ensure!(depth <= 64, "ONNX nesting limit exceeded");
    let mut location = None;
    let mut external = false;
    while let Some(tag) = varint(reader)? {
        let field = tag >> 3;
        anyhow::ensure!(field > 0, "invalid ONNX field number");
        match tag & 7 {
            0 => {
                let value = required_varint(reader)?;
                if kind == "tensor" && field == 14 && value == 1 {
                    external = true;
                }
            }
            1 => discard(reader, 8)?,
            5 => discard(reader, 4)?,
            2 => {
                let size = required_varint(reader)?;
                let mut limited = reader.take(size);
                if kind == "tensor" && field == 13 {
                    anyhow::ensure!(size <= 1 << 20, "oversize ONNX external data entry");
                    let (key, value) = external_entry(&mut limited)?;
                    if key == "location" {
                        anyhow::ensure!(location.is_none(), "duplicate ONNX external location");
                        location = Some(value);
                    }
                } else if let Some(nested) = child(kind, field) {
                    walk(&mut limited, nested, depth + 1, locations)?;
                }
                let remaining = limited.limit();
                discard(&mut limited, remaining)?;
            }
            _ => anyhow::bail!("unsupported ONNX protobuf wire type"),
        }
    }
    if external {
        locations.insert(
            location.ok_or_else(|| anyhow::anyhow!("external ONNX tensor has no location"))?,
        );
    } else if let Some(location) = location {
        // Bind metadata even if a producer omitted data_location; fail closed.
        locations.insert(location);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn field(number: u8, bytes: &[u8]) -> Vec<u8> {
        assert!(bytes.len() < 128);
        [vec![number << 3 | 2, bytes.len() as u8], bytes.to_vec()].concat()
    }
    fn graph(location: &str) -> Vec<u8> {
        let entry = [field(1, b"location"), field(2, location.as_bytes())].concat();
        let tensor = [field(13, &entry), vec![14 << 3, 1]].concat();
        field(7, &field(5, &tensor))
    }
    #[test]
    fn binds_external_content_and_rejects_unsafe_or_missing_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.onnx");
        std::fs::write(&path, graph("weights.bin")).unwrap();
        assert!(capture_onnx(&path).is_err());
        std::fs::write(dir.path().join("weights.bin"), b"old").unwrap();
        let old = capture_onnx(&path).unwrap();
        std::fs::write(dir.path().join("weights.bin"), b"new").unwrap();
        let new = capture_onnx(&path).unwrap();
        assert_eq!(old[0].digest.sha256, new[0].digest.sha256);
        assert_ne!(old[1].digest.sha256, new[1].digest.sha256);
        std::fs::write(&path, graph("../weights.bin")).unwrap();
        assert!(capture_onnx(&path).is_err());
    }
    #[test]
    fn malformed_or_truncated_protobuf_fails_closed() {
        for data in [vec![0], vec![58, 127, 1], vec![58, 1, 128], vec![255; 11]] {
            assert!(walk(&mut data.as_slice(), "model", 0, &mut BTreeSet::new()).is_err());
        }
    }
}

/// Declared graph input metadata, read without materializing tensor payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphInput {
    pub name: String,
    pub element_type: i32,
    pub dimensions: Vec<Option<i64>>,
    pub dimension_symbols: Vec<Option<String>>,
}

enum Field<'a> {
    Integer(u64),
    Bytes(&'a mut dyn Read),
}

fn fields(
    reader: &mut dyn Read,
    visitor: &mut dyn FnMut(u64, Field<'_>) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    while let Some(tag) = varint(reader)? {
        let number = tag >> 3;
        anyhow::ensure!(number > 0, "invalid ONNX field number");
        match tag & 7 {
            0 => visitor(number, Field::Integer(required_varint(reader)?))?,
            1 => discard(reader, 8)?,
            2 => {
                let length = required_varint(reader)?;
                let mut limited = reader.take(length);
                visitor(number, Field::Bytes(&mut limited))?;
                let remaining = limited.limit();
                discard(&mut limited, remaining)?;
            }
            5 => discard(reader, 4)?,
            _ => anyhow::bail!("unsupported ONNX protobuf wire type"),
        }
    }
    Ok(())
}

fn small_string(reader: &mut dyn Read) -> anyhow::Result<String> {
    let mut value = Vec::new();
    reader.take(65537).read_to_end(&mut value)?;
    anyhow::ensure!(value.len() <= 65536, "oversize ONNX input metadata");
    Ok(String::from_utf8(value)?)
}

fn dimension(reader: &mut dyn Read) -> anyhow::Result<(Option<i64>, Option<String>)> {
    let mut result = None;
    let mut symbol = None;
    let mut seen = false;
    fields(reader, &mut |number, value| {
        match (number, value) {
            (1, Field::Integer(value)) => {
                anyhow::ensure!(
                    !seen && value > 0 && value <= i64::MAX as u64,
                    "invalid ONNX input dimension"
                );
                result = Some(value as i64);
                seen = true;
            }
            (2, Field::Bytes(bytes)) => {
                let name = small_string(bytes)?;
                anyhow::ensure!(!seen && !name.is_empty(), "invalid symbolic ONNX dimension");
                symbol = Some(name);
                seen = true;
            }
            _ => {}
        }
        Ok(())
    })?;
    Ok((result, symbol))
}

fn tensor_type(reader: &mut dyn Read, input: &mut GraphInput) -> anyhow::Result<()> {
    let mut has_shape = false;
    fields(reader, &mut |number, value| {
        match (number, value) {
            (1, Field::Integer(value)) => {
                anyhow::ensure!(
                    input.element_type == 0 && value > 0 && value <= i32::MAX as u64,
                    "invalid ONNX input tensor type"
                );
                input.element_type = value as i32;
            }
            (2, Field::Bytes(bytes)) => {
                anyhow::ensure!(!has_shape, "duplicate ONNX input shape");
                has_shape = true;
                fields(bytes, &mut |number, value| {
                    if let (1, Field::Bytes(bytes)) = (number, value) {
                        anyhow::ensure!(input.dimensions.len() < 32, "oversize ONNX input rank");
                        let (size, symbol) = dimension(bytes)?;
                        input.dimensions.push(size);
                        input.dimension_symbols.push(symbol);
                    }
                    Ok(())
                })?;
            }
            _ => {}
        }
        Ok(())
    })?;
    anyhow::ensure!(
        has_shape && input.element_type != 0,
        "incomplete ONNX input type"
    );
    Ok(())
}

fn graph_input(reader: &mut dyn Read) -> anyhow::Result<GraphInput> {
    let mut input = GraphInput {
        name: String::new(),
        element_type: 0,
        dimensions: Vec::new(),
        dimension_symbols: Vec::new(),
    };
    let mut tensor_seen = false;
    fields(reader, &mut |number, value| {
        match (number, value) {
            (1, Field::Bytes(bytes)) => {
                anyhow::ensure!(input.name.is_empty(), "duplicate ONNX input name");
                input.name = small_string(bytes)?;
            }
            (2, Field::Bytes(bytes)) => {
                fields(bytes, &mut |number, value| {
                    match (number, value) {
                        (1, Field::Bytes(bytes)) => {
                            anyhow::ensure!(!tensor_seen, "duplicate ONNX tensor input type");
                            tensor_seen = true;
                            tensor_type(bytes, &mut input)?;
                        }
                        (4 | 5 | 8 | 9, _) => {
                            anyhow::bail!("cached execution requires tensor inputs")
                        }
                        _ => {}
                    }
                    Ok(())
                })?;
            }
            _ => {}
        }
        Ok(())
    })?;
    anyhow::ensure!(
        !input.name.is_empty() && tensor_seen,
        "incomplete ONNX input declaration"
    );
    Ok(input)
}

/// Read the actual top-level inputs using the same bounded ONNX wire reader.
pub fn input_schema(path: &Path) -> anyhow::Result<Vec<GraphInput>> {
    let mut inputs = Vec::new();
    let mut graph_seen = false;
    fields(
        &mut BufReader::new(File::open(path)?),
        &mut |number, value| {
            if let (7, Field::Bytes(bytes)) = (number, value) {
                anyhow::ensure!(!graph_seen, "duplicate ONNX model graph");
                graph_seen = true;
                fields(bytes, &mut |number, value| {
                    if let (11, Field::Bytes(bytes)) = (number, value) {
                        anyhow::ensure!(inputs.len() < 256, "oversize ONNX input count");
                        let input = graph_input(bytes)?;
                        anyhow::ensure!(
                            !inputs.iter().any(|old: &GraphInput| old.name == input.name),
                            "duplicate ONNX input name"
                        );
                        inputs.push(input);
                    }
                    Ok(())
                })?;
            }
            Ok(())
        },
    )?;
    anyhow::ensure!(
        graph_seen && !inputs.is_empty(),
        "ONNX graph has no declared inputs"
    );
    Ok(inputs)
}
