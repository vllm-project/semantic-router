package config

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestGeneratedCanonicalContractSchemaIsCurrent(t *testing.T) {
	t.Parallel()

	_, sourceFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve contract schema test path")
	}
	schemaPath := filepath.Clean(filepath.Join(
		filepath.Dir(sourceFile),
		"..",
		"..",
		"..",
		"vllm-sr",
		"cli",
		"templates",
		"canonical-config-schema.json",
	))
	committed, err := os.ReadFile(schemaPath)
	if err != nil {
		t.Fatalf("read generated canonical config schema: %v", err)
	}
	generated, err := json.MarshalIndent(BuildCanonicalContractSchema(), "", "  ")
	if err != nil {
		t.Fatalf("marshal canonical config schema: %v", err)
	}
	generated = append(generated, '\n')
	if !bytes.Equal(committed, generated) {
		t.Fatalf(
			"%s is stale; regenerate with `go run ./cmd/config-contract-schema > ../vllm-sr/cli/templates/canonical-config-schema.json`",
			schemaPath,
		)
	}
}
