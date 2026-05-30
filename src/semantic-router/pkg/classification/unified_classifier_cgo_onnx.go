//go:build onnx && !windows && cgo

package classification

// Link against onnx-binding Rust library.
// Build with: go build -tags=onnx

/*
#cgo LDFLAGS: -L../../../../../onnx-binding/target/release -lonnx_semantic_router
*/
import "C"

var nativeBackendCapabilities = NativeBackendCapabilities{
	Name:                       "onnx",
	UnifiedBatchClassification: false,
	LoRABatchClassification:    false,
	BatchedEmbedding:           true,
	MultimodalEmbedding:        false,
	ModalityRouting:            false,
	MLPSelector:                false,
	ExplicitReset:              false,
}
