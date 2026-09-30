//go:build onnx

package config

// Preserve the historical onnx build-tag default for source-build callers.
var defaultModelProvider = "ort"
