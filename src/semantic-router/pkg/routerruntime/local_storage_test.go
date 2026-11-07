package routerruntime

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func vectorStoreAt(dir string) *config.RouterConfig {
	return &config.RouterConfig{VectorStore: &config.VectorStoreConfig{Enabled: true, FileStorageDir: dir}}
}

func TestCheckLocalStorageAcceptsWritableDirsAndCreatesNothing(t *testing.T) {
	root := t.TempDir()
	for _, dir := range []string{root, filepath.Join(root, "data", "files")} {
		if err := CheckLocalStorage(vectorStoreAt(dir)); err != nil {
			t.Fatalf("%s: %v", dir, err)
		}
	}
	entries, err := os.ReadDir(root)
	if err != nil || len(entries) != 0 {
		t.Fatalf("the check left %v behind (%v)", entries, err)
	}
}

func TestCheckLocalStorageNamesTheSettingOfAnUnwritableDir(t *testing.T) {
	file := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	readOnly := filepath.Join(t.TempDir(), "read-only")
	if err := os.Mkdir(readOnly, 0o500); err != nil {
		t.Fatal(err)
	}
	tests := map[string]string{
		file:                                "is not a directory",
		filepath.Join(file, "data"):         "not a directory",
		filepath.Join(readOnly, "vsr/data"): "permission denied",
	}
	for dir, cause := range tests {
		if cause == "permission denied" && os.Geteuid() == 0 {
			continue // root writes through permission bits
		}
		err := CheckLocalStorage(vectorStoreAt(dir))
		if err == nil {
			t.Fatalf("%s: the check passed", dir)
		}
		for _, want := range []string{
			"global.stores.vector_store.file_storage_dir: " + dir + " is not writable",
			cause,
			"mount a writable volume at that path, or set global.stores.vector_store.file_storage_dir to a writable directory",
		} {
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("%s: error %q lacks %q", dir, err, want)
			}
		}
	}
}

func TestUnwritableDirsReportsEveryDirectory(t *testing.T) {
	file := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := vectorStoreAt(filepath.Join(file, "files"))
	cfg.Looper.Flow.State.File.Directory = filepath.Join(file, "flow")
	cfg.Decisions = []config.Decision{{Name: "flow", Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmWorkflows}}}
	unwritable := UnwritableDirs(cfg)
	if len(unwritable) != 2 ||
		unwritable[0].Setting != "global.stores.vector_store.file_storage_dir" ||
		unwritable[1].Setting != "global.integrations.looper.flow.state.file.directory" {
		t.Fatalf("UnwritableDirs() = %+v", unwritable)
	}
	if err := CheckLocalStorage(cfg); err == nil || strings.Count(err.Error(), "is not writable") != 2 {
		t.Fatalf("CheckLocalStorage() = %v, want both directories", err)
	}
}
