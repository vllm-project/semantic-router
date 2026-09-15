package recipe

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestSourceRecipeRuntimeBinding(t *testing.T) {
	directory := copyMaintainedRecipe(t, "accuracy")
	sourceConfig, err := os.ReadFile(filepath.Join(directory, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name       string
		runtime    []byte
		provenance func([]byte) string
		ready      bool
	}{
		{name: "identical_without_provenance", runtime: sourceConfig, ready: true},
		{
			name:    "realized_with_exact_provenance",
			runtime: append(append([]byte(nil), sourceConfig...), []byte("\n# platform realization\n")...),
			provenance: func(runtime []byte) string {
				return runtimeProvenanceJSON(digestBytes(sourceConfig), digestBytes(runtime))
			},
			ready: true,
		},
		{name: "drift_without_provenance", runtime: append(sourceConfig, []byte("\n# drift\n")...)},
		{name: "malformed_provenance", runtime: append(sourceConfig, []byte("\n# realized\n")...), provenance: func([]byte) string { return "{" }},
		{
			name:    "trailing_provenance_document",
			runtime: append(sourceConfig, []byte("\n# realized\n")...),
			provenance: func(runtime []byte) string {
				return runtimeProvenanceJSON(digestBytes(sourceConfig), digestBytes(runtime)) + "{}"
			},
		},
		{
			name:    "source_digest_mismatch",
			runtime: append(sourceConfig, []byte("\n# realized\n")...),
			provenance: func(runtime []byte) string {
				return runtimeProvenanceJSON(digestBytes([]byte("different source")), digestBytes(runtime))
			},
		},
		{
			name:    "runtime_digest_mismatch",
			runtime: append(sourceConfig, []byte("\n# realized\n")...),
			provenance: func([]byte) string {
				return runtimeProvenanceJSON(digestBytes(sourceConfig), digestBytes([]byte("different runtime")))
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			runtimePath := filepath.Join(root, "runtime-config.stack.yaml")
			if err := os.WriteFile(runtimePath, test.runtime, 0o600); err != nil {
				t.Fatal(err)
			}
			if test.provenance != nil {
				provenancePath := filepath.Join(root, "runtime-config.stack.provenance.json")
				if err := os.WriteFile(provenancePath, []byte(test.provenance(test.runtime)), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			service := NewService(Options{Directory: directory, RuntimeConfigPath: runtimePath})
			descriptor, err := service.Describe()
			if err != nil {
				t.Fatalf("Describe(): %v", err)
			}
			if test.ready {
				if !descriptor.Managed || descriptor.SourceHealth.Status != "ready" {
					t.Fatalf("descriptor = %#v", descriptor)
				}
				return
			}
			if descriptor.Managed || descriptor.SourceHealth.Status != "unmanaged" {
				t.Fatalf("descriptor = %#v", descriptor)
			}
			if _, err := service.ListProbes(ListOptions{}); !errors.Is(err, ErrUnmanaged) {
				t.Fatalf("ListProbes() error = %v, want ErrUnmanaged", err)
			}
		})
	}
}

func TestSourceRecipeMissingRuntimeFailsClosed(t *testing.T) {
	directory := copyMaintainedRecipe(t, "accuracy")
	service := NewService(Options{
		Directory:         directory,
		RuntimeConfigPath: filepath.Join(t.TempDir(), "missing.yaml"),
	})
	descriptor, err := service.Describe()
	if err != nil || descriptor.Managed || descriptor.SourceHealth.Status != "unmanaged" {
		t.Fatalf("Describe() = %#v, %v", descriptor, err)
	}
	if _, err := service.ListProbes(ListOptions{}); !errors.Is(err, ErrUnmanaged) {
		t.Fatalf("ListProbes() error = %v, want ErrUnmanaged", err)
	}
}

func runtimeProvenanceJSON(sourceDigest, runtimeDigest string) string {
	return fmt.Sprintf(
		`{"schema_version":%q,"source_digest":%q,"last_materialized_active_digest":%q}`,
		runtimeConfigProvenanceSchema,
		sourceDigest,
		runtimeDigest,
	)
}
