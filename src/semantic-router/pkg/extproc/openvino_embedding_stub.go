//go:build !openvino || windows || !cgo

package extproc

import "fmt"

func openvinoEmbeddingFunc(_ string) func(string) ([]float32, error) {
	return func(_ string) ([]float32, error) {
		return nil, fmt.Errorf("openvino backend not available: binary built without openvino tag")
	}
}
