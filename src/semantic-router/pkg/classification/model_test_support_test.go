package classification

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	. "github.com/onsi/ginkgo/v2"
)

var testModelWeightCandidates = []string{
	"model.safetensors",
	"model.safetensors.index.json",
	"pytorch_model.bin",
	"adapter_model.safetensors",
}

func testModelArtifactsAvailable(modelPath string) bool {
	info, err := os.Stat(modelPath)
	if err != nil || !info.IsDir() {
		return false
	}
	if _, err := os.Stat(filepath.Join(modelPath, "config.json")); err != nil {
		return false
	}
	for _, candidate := range testModelWeightCandidates {
		if _, err := os.Stat(filepath.Join(modelPath, candidate)); err == nil {
			return true
		}
	}
	if matches, _ := filepath.Glob(filepath.Join(modelPath, "*.safetensors")); len(matches) > 0 {
		return true
	}
	if matches, _ := filepath.Glob(filepath.Join(modelPath, "*.bin")); len(matches) > 0 {
		return true
	}
	return false
}

func skipTestIfModelArtifactsMissing(t *testing.T, label string, modelPath string) {
	t.Helper()
	if testModelArtifactsAvailable(modelPath) {
		return
	}
	t.Skipf("%s artifacts not available at %s (missing model weights)", label, modelPath)
}

func skipSpecIfModelArtifactsMissing(label string, modelPath string) {
	if testModelArtifactsAvailable(modelPath) {
		return
	}
	Skip(fmt.Sprintf("%s artifacts not available at %s (missing model weights)", label, modelPath))
}
