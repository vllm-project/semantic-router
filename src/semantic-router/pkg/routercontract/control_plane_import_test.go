package routercontract

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestControlPlanePackagesUseRouterContractSeams(t *testing.T) {
	repoRoot := controlPlaneRepoRoot(t)
	for _, relDir := range []string{
		filepath.Join("dashboard", "backend"),
		filepath.Join("deploy", "operator", "controllers"),
	} {
		assertNoDirectImport(t, filepath.Join(repoRoot, relDir), "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config")
		assertNoDirectImport(t, filepath.Join(repoRoot, relDir), "github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl")
	}
}

func controlPlaneRepoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", "..", ".."))
}

func assertNoDirectImport(t *testing.T, root string, forbidden string) {
	t.Helper()
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() || filepath.Ext(path) != ".go" {
			return nil
		}

		content, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if strings.Contains(string(content), forbidden) {
			t.Errorf("%s still imports forbidden control-plane dependency %q", path, forbidden)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", root, err)
	}
}
