package gateway

import (
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
)

func TestSemanticRouterInstallOptionsUsesBaseValuesFileByDefault(t *testing.T) {
	stack := New(Config{
		Name:                     "unit-test",
		SemanticRouterValuesFile: "base-values.yaml",
	})

	opts := stack.semanticRouterInstallOptions(&framework.SetupOptions{
		ImageTag: "test-image",
	})

	if len(opts.ValuesFiles) != 1 || opts.ValuesFiles[0] != "base-values.yaml" {
		t.Fatalf("expected only the base values file, got %#v", opts.ValuesFiles)
	}
}

func TestSemanticRouterInstallOptionsAppendsWorkspaceOverlay(t *testing.T) {
	stack := New(Config{
		Name:                     "unit-test",
		SemanticRouterValuesFile: "base-values.yaml",
	})

	opts := stack.semanticRouterInstallOptions(&framework.SetupOptions{
		ImageTag: "test-image",
		ValuesFiles: map[string]string{
			releaseSemanticRouter: "workspace-models.yaml",
		},
	})

	if len(opts.ValuesFiles) != 2 {
		t.Fatalf("expected base values plus workspace overlay, got %#v", opts.ValuesFiles)
	}
	if opts.ValuesFiles[0] != "base-values.yaml" || opts.ValuesFiles[1] != "workspace-models.yaml" {
		t.Fatalf("unexpected values file order: %#v", opts.ValuesFiles)
	}
}
