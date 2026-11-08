package config

import (
	"strings"
	"testing"
)

func TestRouterDeploymentsLeaveModelDownloadsToTheModelRuntime(t *testing.T) {
	// The model runtime downloads and verifies the models it serves, so the
	// init container only prepares the cache directories.
	for _, asset := range []string{"deploy/kserve/deployment.yaml", "deploy/openshift/deployment.yaml"} {
		t.Run(asset, func(t *testing.T) {
			data := string(mustReadRepoFile(t, asset))
			if !strings.Contains(data, "name: prepare-cache") {
				t.Fatal("deployment has no cache preparation init container")
			}
			for _, preload := range []string{
				"snapshot_download",
				"vllm-sr/Vela-",
				"mmbert32k-jailbreak-detector-merged",
				"Encoder-307M-Embedding",
				"EMBEDDING_MODEL",
			} {
				if strings.Contains(data, preload) {
					t.Fatalf("deployment preloads a model the model runtime fetches: found %q", preload)
				}
			}
		})
	}
}
