//go:build windows || !cgo || (!amd64 && !arm64)

package onnx_binding

// EmbeddingCapabilitiesFor fails closed when the native ONNX runtime cannot
// be linked into this build. It never returns plausible static metadata for an
// unavailable backend.
func EmbeddingCapabilitiesFor(string) (EmbeddingCapabilities, error) {
	return EmbeddingCapabilities{}, ErrBackendUnavailable
}
