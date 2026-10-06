package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configschema"
)

func main() {
	repositoryRoot := flag.String("repository-root", "", "path to the semantic-router repository root")
	check := flag.Bool("check", false, "verify generated schema artifacts without rewriting them")
	flag.Parse()

	root, err := filepath.Abs(*repositoryRoot)
	if err != nil || *repositoryRoot == "" {
		fatalf("--repository-root is required")
	}
	payload, err := configschema.GenerateFromSource(root)
	if err != nil {
		fatalf("generate schema: %v", err)
	}
	typeScriptPayload, err := configschema.GenerateTypeScriptContract(payload)
	if err != nil {
		fatalf("generate TypeScript contract: %v", err)
	}

	outputs := []struct {
		path    string
		payload []byte
	}{
		{filepath.Join(root, "src", "semantic-router", "pkg", "configschema", "router-config-v0.3.schema.json"), payload},
		{filepath.Join(root, "dashboard", "frontend", "src", "generated", "routerConfigContract.ts"), typeScriptPayload},
	}
	for _, output := range outputs {
		if *check {
			current, readErr := os.ReadFile(output.path)
			if readErr != nil || string(current) != string(output.payload) {
				fatalf("generated config contract is stale: %s", output.path)
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(output.path), 0o755); err != nil {
			fatalf("create output directory: %v", err)
		}
		if err := os.WriteFile(output.path, output.payload, 0o644); err != nil {
			fatalf("write %s: %v", output.path, err)
		}
	}
}

func fatalf(format string, values ...any) {
	_, _ = fmt.Fprintf(os.Stderr, format+"\n", values...)
	os.Exit(1)
}
