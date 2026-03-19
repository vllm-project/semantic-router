package modelresearch

import (
	"context"
	"testing"
)

func TestPythonCUDAAvailableDetectsUnavailableGPU(t *testing.T) {
	t.Parallel()

	manager, err := NewManager(ManagerConfig{
		BaseDir:        t.TempDir(),
		RepoRoot:       t.TempDir(),
		PythonPath:     "python3",
		DefaultAPIBase: "http://router.internal",
		CommandRunner: func(_ context.Context, _ commandSpec, onLine func(stream, line string)) error {
			onLine("stdout", "0")
			return nil
		},
	})
	if err != nil {
		t.Fatalf("NewManager() error = %v", err)
	}

	available, err := manager.pythonCUDAAvailable(context.Background())
	if err != nil {
		t.Fatalf("pythonCUDAAvailable() error = %v", err)
	}
	if available {
		t.Fatalf("pythonCUDAAvailable() = true, want false")
	}
}
