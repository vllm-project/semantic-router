package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// generatedFile is a checked-in file rendered from the header registry.
type generatedFile struct {
	// path is relative to the repository root.
	path   string
	render func() ([]byte, error)
}

// generatedFiles are the reference the header API reference page is rendered
// from, and the Dashboard's constants.
var generatedFiles = []generatedFile{
	{path: "src/semantic-router/pkg/headers/reference.generated.json", render: func() ([]byte, error) {
		return headers.NewReference(config.SupportedDebugHeadersModes()).JSON()
	}},
	{path: "dashboard/frontend/src/generated/vsrHeaders.ts", render: func() ([]byte, error) {
		return []byte(headers.TypeScriptConstants()), nil
	}},
}

func main() {
	repositoryRoot := flag.String("repository-root", "", "path to the semantic-router repository root")
	check := flag.Bool("check", false, "verify the generated header references without rewriting them")
	flag.Parse()

	root, err := filepath.Abs(*repositoryRoot)
	if err != nil || *repositoryRoot == "" {
		fatalf("--repository-root is required")
	}
	for _, file := range generatedFiles {
		path := filepath.Join(root, filepath.FromSlash(file.path))
		payload, err := file.render()
		if err != nil {
			fatalf("render %s: %v", path, err)
		}
		if *check {
			current, err := os.ReadFile(path)
			if err != nil {
				fatalf("read %s: %v", path, err)
			}
			if !bytes.Equal(current, payload) {
				fatalf("generated header reference is stale: %s; run make headers-generate", path)
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			fatalf("create output directory: %v", err)
		}
		if err := os.WriteFile(path, payload, 0o644); err != nil {
			fatalf("write %s: %v", path, err)
		}
	}
}

func fatalf(format string, values ...any) {
	_, _ = fmt.Fprintf(os.Stderr, format+"\n", values...)
	os.Exit(1)
}
