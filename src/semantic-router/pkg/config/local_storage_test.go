package config

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// The reference config turns on both writers, so its settings are the keys a
// configuration really writes.
func TestReferenceConfigLocalStorageDirsNameTheirSettings(t *testing.T) {
	cfg, err := Parse(filepath.Join(referenceConfigRepoRoot(t), "config", "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	want := []LocalStorageDir{
		{Setting: "global.stores.vector_store.file_storage_dir", Path: "/var/lib/vsr/data"},
		{Setting: "global.integrations.looper.flow.state.file.directory", Path: ".vllm-sr/flow-state"},
	}
	if got := cfg.LocalStorageDirs(); !reflect.DeepEqual(got, want) {
		t.Fatalf("LocalStorageDirs() = %+v, want %+v", got, want)
	}
}

func TestLocalStorageDirsFollowTheFeaturesThatWrite(t *testing.T) {
	flowDecision := Decision{Name: "flow", Algorithm: &AlgorithmConfig{Type: DecisionAlgorithmWorkflows}}
	flowDir := LocalStorageDir{Setting: "global.integrations.looper.flow.state.file.directory", Path: "/srv/flow"}
	tests := []struct {
		name string
		cfg  RouterConfig
		want []LocalStorageDir
	}{
		{name: "nothing writes", cfg: RouterConfig{}},
		{name: "vector store disabled", cfg: RouterConfig{VectorStore: &VectorStoreConfig{FileStorageDir: "/srv/files"}}},
		{
			name: "vector store default",
			cfg:  RouterConfig{VectorStore: &VectorStoreConfig{Enabled: true}},
			want: []LocalStorageDir{{Setting: "global.stores.vector_store.file_storage_dir", Path: DefaultVectorStoreFileStorageDir}},
		},
		{
			name: "vector store directory",
			cfg:  RouterConfig{VectorStore: &VectorStoreConfig{Enabled: true, FileStorageDir: "/srv/files"}},
			want: []LocalStorageDir{{Setting: "global.stores.vector_store.file_storage_dir", Path: "/srv/files"}},
		},
		{
			name: "flow directory with the default file backend",
			cfg:  flowConfig(WorkflowStateRuntimeConfig{File: WorkflowStateFileConfig{Directory: "/srv/flow"}}, flowDecision),
			want: []LocalStorageDir{flowDir},
		},
		{
			name: "flow directory without a workflows decision",
			cfg:  flowConfig(WorkflowStateRuntimeConfig{File: WorkflowStateFileConfig{Directory: "/srv/flow"}}),
		},
		{
			name: "flow directory under another backend",
			cfg: flowConfig(WorkflowStateRuntimeConfig{
				StoreBackend: WorkflowStateBackendRedis, File: WorkflowStateFileConfig{Directory: "/srv/flow"},
			}, flowDecision),
		},
		{name: "flow state in the user cache directory", cfg: flowConfig(WorkflowStateRuntimeConfig{}, flowDecision)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.cfg.LocalStorageDirs(); !reflect.DeepEqual(got, test.want) {
				t.Fatalf("LocalStorageDirs() = %+v, want %+v", got, test.want)
			}
		})
	}
}

// The storage pages tell users to keep local storage in the Router's data
// directory, which the Helm chart mounts writable, so every Router example on
// them must write there.
func TestStorageDocExamplesWriteInTheDataDirectory(t *testing.T) {
	for _, page := range routerCheckedDocPages {
		if filepath.Base(page) != "storage-overview.md" {
			continue
		}
		t.Run(page, func(t *testing.T) {
			content, err := os.ReadFile(filepath.Join(referenceConfigRepoRoot(t), page))
			if err != nil {
				t.Fatal(err)
			}
			checked := 0
			for _, fence := range docFences(string(content)) {
				if fence.lang != "yaml" || fence.helmValues() {
					continue
				}
				cfg, err := ParseYAMLBytes(docExampleDocument(t, fence.body))
				if err != nil {
					t.Fatalf("%s:%d: %v", page, fence.line, err)
				}
				dirs := cfg.LocalStorageDirs()
				if len(dirs) == 0 {
					t.Errorf("%s:%d: the example writes no local directory", page, fence.line)
				}
				for _, dir := range dirs {
					if !strings.HasPrefix(dir.Path, "/var/lib/vsr/") {
						t.Errorf("%s:%d: %s is %s, outside /var/lib/vsr", page, fence.line, dir.Setting, dir.Path)
					}
				}
				checked++
			}
			if checked == 0 {
				t.Fatalf("%s has no Router example", page)
			}
		})
	}
}

func flowConfig(state WorkflowStateRuntimeConfig, decisions ...Decision) RouterConfig {
	cfg := RouterConfig{}
	cfg.Looper.Flow.State = state
	cfg.Decisions = decisions
	return cfg
}
