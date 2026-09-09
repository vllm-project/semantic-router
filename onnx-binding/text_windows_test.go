//go:build !windows && cgo && (amd64 || arm64)

package onnx_binding

import (
	"math"
	"os"
	"os/exec"
	"testing"
)

func TestTextWindowsWithoutModel(t *testing.T) {
	const childEnv = "SR_ONNX_TEXT_WINDOWS_CHILD"
	if os.Getenv(childEnv) != "1" {
		cmd := exec.Command(os.Args[0], "-test.run=^TestTextWindowsWithoutModel$")
		cmd.Env = append(os.Environ(), childEnv+"=1")
		if output, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("uninitialized binding: %v\n%s", err, output)
		}
		return
	}
	for _, text := range []string{"query", "", "你好"} {
		for _, limit := range []int{0, -1, 32} {
			windows, err := TextWindows(text, limit)
			if err == nil || windows != nil {
				t.Fatalf("TextWindows(%q, %d) = %v, %v; want nil, error", text, limit, windows, err)
			}
		}
	}
}

func TestTextWindowsInvalidInput(t *testing.T) {
	for _, input := range []struct {
		text  string
		limit int
	}{
		{"before\x00after", 0},
		{"\xff", 0},
		{"query", math.MaxInt32 + 1},
	} {
		if windows, err := TextWindows(input.text, input.limit); err == nil || windows != nil {
			t.Fatalf("TextWindows(%q, %d) = %v, %v; want nil, error", input.text, input.limit, windows, err)
		}
	}
}
