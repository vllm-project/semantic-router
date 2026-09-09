package onnx_binding

import (
	"os"
	"os/exec"
	"testing"
)

// Match the router's replacement-module API, including its concrete range type.
var _ func(string, int) ([]TextWindow, error) = TextWindows

func TestTextWindowsReportsUninitializedModel(t *testing.T) {
	if os.Getenv("ONNX_TEXT_WINDOWS_TEST_CHILD") != "1" {
		// Other binding tests may initialize the process-wide model singleton.
		executable, err := os.Executable()
		if err != nil {
			t.Fatalf("resolve test executable: %v", err)
		}
		command := exec.Command(executable, "-test.run=^TestTextWindowsReportsUninitializedModel$")
		command.Env = append(os.Environ(), "ONNX_TEXT_WINDOWS_TEST_CHILD=1")
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("uninitialized model check: %v\n%s", err, output)
		}
		return
	}
	windows, err := TextWindows("a query whose tail must not be silently discarded", 0)
	if err == nil || windows != nil {
		t.Fatalf("TextWindows without a model = %v, %v; want nil and an error", windows, err)
	}
	outputs, err := GetEmbeddingsBatch([]string{"query"}, 0, 0)
	if err == nil || outputs != nil {
		t.Fatalf("GetEmbeddingsBatch without a model = %v, %v; want nil and an error", outputs, err)
	}
}

func TestGetEmbeddingsBatchRejectsEmptyInput(t *testing.T) {
	if _, err := GetEmbeddingsBatch(nil, 0, 0); err == nil {
		t.Fatal("GetEmbeddingsBatch must reject an empty batch")
	}
}
