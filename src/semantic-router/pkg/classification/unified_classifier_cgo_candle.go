//go:build !onnx && !windows && cgo

package classification

// Link against candle-binding Rust library (default).
// To use onnx-binding instead, build with: go build -tags=onnx

/*
#cgo LDFLAGS: -L../../../../../candle-binding/target/release -lcandle_semantic_router
*/
import "C"
