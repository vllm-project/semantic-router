package classification

import (
	"os"
	"testing"
)

// Legacy discovery compatibility is a manual opt-in. Published-model CI uses
// explicit registered family paths, never a scan of the ambient model cache.
func requireLegacyTestModelsDir(t *testing.T) string {
	t.Helper()
	const variable = "SEMANTIC_ROUTER_TEST_MODELS_DIR"
	path := os.Getenv(variable)
	if path == "" {
		t.Skip("set SEMANTIC_ROUTER_TEST_MODELS_DIR to exercise legacy model discovery")
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("explicit legacy model directory %q: %v", path, err)
	}
	if !info.IsDir() {
		t.Fatalf("explicit legacy model path is not a directory: %s", path)
	}
	return path
}
