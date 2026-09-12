//go:build windows || !cgo || (!amd64 && !arm64)

package onnx_binding

import "fmt"

type SequenceModel struct{}

func OpenSequenceModel(SequenceModelOptions) (*SequenceModel, error) {
	return nil, fmt.Errorf("native sequence models are unavailable on this build")
}
func (*SequenceModel) Classify(string) ([]float32, error) {
	return nil, fmt.Errorf("native sequence models are unavailable on this build")
}
func (*SequenceModel) Close() error { return nil }
