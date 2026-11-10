package config

import (
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// Every ext_proc filter the repository ships must let the Router set exactly
// the per-request reliability headers, so a decision's reliability block works
// behind every Envoy the repository installs, and ext_proc may change no other
// x-envoy-* header.
func TestEveryShippedExtProcFilterAllowsExactlyTheReliabilityHeaders(t *testing.T) {
	root := repoRootFromTestFile(t)
	names := make([]string, len(ReliabilityHeaders))
	for i, header := range ReliabilityHeaders {
		names[i] = strings.TrimPrefix(header, "x-envoy-")
	}
	rule := `regex: "^x-envoy-(` + strings.Join(names, "|") + `)$"`

	repo := os.DirFS(root)
	var checked []string
	for _, dir := range []string{"deploy", "e2e", "src/vllm-sr/cli/templates"} {
		err := fs.WalkDir(repo, dir, func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				if entry.Name() == "node_modules" {
					return fs.SkipDir
				}
				return nil
			}
			if !slices.Contains([]string{".yaml", ".yml", ".go", ".json"}, filepath.Ext(path)) {
				return nil
			}
			data, err := fs.ReadFile(repo, path)
			if err != nil {
				return err
			}
			text := string(data)
			filters := strings.Count(text, "envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor")
			if filters == 0 {
				return nil
			}
			checked = append(checked, path)
			if got := strings.Count(text, rule); got != filters {
				t.Errorf("%s: %d ext_proc filters, %d with the reliability header rule", path, filters, got)
			}
			if strings.Contains(text, "allow_envoy") || strings.Count(text, "allow_expression") != filters {
				t.Errorf("%s: ext_proc mutation rules may allow only the reliability headers", path)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if !slices.Contains(checked, "src/vllm-sr/cli/templates/envoy.template.yaml") {
		t.Fatalf("checked %v, want the CLI's Envoy template among them", checked)
	}
}
