package gateway

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
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

func TestDeferredSemanticRouterReadinessDoesNotBlockHelmInstall(t *testing.T) {
	stack := New(Config{
		Name:                         "managed-bootstrap",
		SemanticRouterValuesFile:     "managed-values.yaml",
		DeferSemanticRouterReadiness: true,
	})

	opts := stack.semanticRouterInstallOptions(&framework.SetupOptions{ImageTag: "test-image"})
	if opts.Wait {
		t.Fatal("managed bootstrap must install semantic-router without --wait")
	}
	if opts.Timeout != "" {
		t.Fatalf("non-waiting semantic-router install timeout = %q, want empty", opts.Timeout)
	}
}

func TestDeferredSemanticRouterReadinessRemovesOnlyRouterFromEarlyVerification(t *testing.T) {
	stack := New(Config{
		Name:                         "managed-bootstrap",
		DeferSemanticRouterReadiness: true,
	})

	want := []helpers.DeploymentRef{
		{Namespace: helm.EnvoyGatewayRelease.Namespace, Name: deploymentEnvoyGateway},
		{Namespace: helm.AIGatewayRelease.Namespace, Name: deploymentAIGateway},
	}
	if !slices.Equal(stack.config.VerifyDeployments, want) {
		t.Fatalf("early verification deployments = %#v, want %#v", stack.config.VerifyDeployments, want)
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
			helm.SemanticRouterRelease.ReleaseName: "workspace-models.yaml",
		},
	})

	if len(opts.ValuesFiles) != 2 {
		t.Fatalf("expected base values plus workspace overlay, got %#v", opts.ValuesFiles)
	}
	if opts.ValuesFiles[0] != "base-values.yaml" || opts.ValuesFiles[1] != "workspace-models.yaml" {
		t.Fatalf("unexpected values file order: %#v", opts.ValuesFiles)
	}
}

// Teardown must mirror setup in reverse. Prerequisites are applied before the
// Helm releases, so they have to be removed after them: a profile may create
// the release's own namespace as a prerequisite, and deleting that namespace
// before `helm uninstall` takes the release secret with it, leaving the
// uninstall a no-op and the chart's cluster-scoped RBAC orphaned.
func TestTeardownRemovesPrerequisitesAfterTheReleasesThatUseThem(t *testing.T) {
	stack := New(Config{Name: "unit-test"})

	var order []string
	for _, step := range stack.teardownSteps() {
		order = append(order, step.name)
	}

	want := []string{teardownStepResources, teardownStepCoreReleases, teardownStepPrerequisites}
	if len(order) != len(want) {
		t.Fatalf("teardown steps = %v, want %v", order, want)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("teardown steps = %v, want %v", order, want)
		}
	}
}
