package config

import (
	"regexp"
	"strings"
	"testing"
)

func TestVelaDeploymentDownloadsMatchRegistryPinsAndPaths(t *testing.T) {
	pattern := regexp.MustCompile(`\("(vllm-sr/Vela-[^"]+)", "(Vela-[^"]+)", "([a-f0-9]{40})"\)`)
	// OpenShift's provider fixtures share one Router deployment and its pinned preloads.
	for _, asset := range []string{"deploy/kserve/deployment.yaml", "deploy/openshift/deployment.yaml"} {
		t.Run(asset, func(t *testing.T) {
			data := string(mustReadRepoFile(t, asset))
			matches := pattern.FindAllStringSubmatch(data, -1)
			if len(matches) != 3 {
				t.Fatalf("found %d pinned preloads, want 3", len(matches))
			}
			if !strings.Contains(data, "Vela-1.0-Encoder-307M-Guard") || strings.Contains(data, "mmbert32k-jailbreak-detector-merged") {
				t.Fatal("deployment still preloads the old default Guard")
			}
			// The model runtime fetches the embedding deployment's artifact.
			if strings.Contains(data, "Encoder-307M-Embedding") || strings.Contains(data, "EMBEDDING_MODEL") {
				t.Fatal("deployment preloads an embedding model")
			}
			for _, match := range matches {
				spec := GetModelByPath("models/" + match[2])
				if spec == nil || spec.RepoID != match[1] || spec.Revision != match[3] {
					t.Fatalf("preload identity disagrees with Router registry: %v", match)
				}
			}
			if !strings.Contains(data, "revision=revision") || !strings.Contains(data, `if not has_model or revision != "main":`) {
				t.Fatal("immutable preload can be bypassed by old file presence")
			}
			if !strings.Contains(data, `ignore_patterns=["reproduction/*", "reproducibility/*", "lora/*"]`) {
				t.Fatal("optional training artifacts would be preloaded")
			}
		})
	}
}
