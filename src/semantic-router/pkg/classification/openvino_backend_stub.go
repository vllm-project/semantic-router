//go:build !openvino || windows || !cgo

package classification

import (
	"fmt"
	"os"
	"strings"
)

// embeddingBackendOverride reads the EMBEDDING_BACKEND_OVERRIDE env var (normalized).
func embeddingBackendOverride() string {
	return strings.ToLower(strings.TrimSpace(os.Getenv("EMBEDDING_BACKEND_OVERRIDE")))
}

func initOpenVINOModel(modelType, mmBertModelPath, fallbackModelPath string, useCPU bool) error {
	return fmt.Errorf("openvino backend requires non-windows build with cgo enabled")
}

func getOpenVINOEmbedding(modelType, text string, targetDim int) ([]float32, error) {
	return nil, fmt.Errorf("openvino backend requires non-windows build with cgo enabled")
}

// openvinoClassResult mirrors openvino_binding.ClassResult for non-cgo builds.
type openvinoClassResult struct {
	Class      int
	Confidence float32
}

func initOpenVINOClassifier(modelPath string, numClasses int, useCPU bool) error {
	return fmt.Errorf("openvino backend requires non-windows build with cgo enabled")
}

func classifyOpenVINO(text string) (openvinoClassResult, error) {
	return openvinoClassResult{}, fmt.Errorf("openvino backend requires non-windows build with cgo enabled")
}
