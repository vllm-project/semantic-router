package mlmodelselection

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestModelCopiesTargetSelectedBatchCluster(t *testing.T) {
	directory := t.TempDir()
	capture := filepath.Join(directory, "arguments")
	t.Setenv("CAPTURE_KIND_ARGUMENTS", capture)
	t.Setenv("PATH", directory+string(os.PathListSeparator)+os.Getenv("PATH"))
	script := "#!/bin/sh\nprintf '%s\\n' \"$*\" > \"$CAPTURE_KIND_ARGUMENTS\"\nprintf '%s\\n' isolated-worker\n"
	if err := os.WriteFile(filepath.Join(directory, "kind"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	nodes, err := NewProfile().getKindNodes(context.Background(), "isolated-profile")
	if err != nil || len(nodes) != 1 || nodes[0] != "isolated-worker" {
		t.Fatalf("get nodes: %v, %v", nodes, err)
	}
	arguments, err := os.ReadFile(capture)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(arguments)) != "get nodes --name isolated-profile" {
		t.Fatalf("wrong cluster selected: %s", arguments)
	}
}
